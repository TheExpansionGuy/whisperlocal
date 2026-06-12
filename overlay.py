"""WhisperLocal overlay — animated pill with word-by-word transcript, state morph, spring resize."""
import math
import time
import objc
from AppKit import (
    NSBorderlessWindowMask, NSBezierPath, NSColor, NSFont,
    NSMakeRect, NSNonactivatingPanelMask, NSPanel, NSScreen, NSView,
    NSForegroundColorAttributeName, NSFontAttributeName,
    NSParagraphStyleAttributeName, NSMutableParagraphStyle,
    NSTextAlignmentCenter, NSTextAlignmentLeft, NSTextAlignmentRight,
    NSLineBreakByTruncatingTail, NSLineBreakByWordWrapping,
    NSAnimationContext,
)
from Foundation import NSMakePoint, NSObject, NSTimer, NSString

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
PANEL_W_COMPACT = 290    # controls-only width (idle / recording, no text yet)
PANEL_W_WIDE    = 540    # max width when showing transcript
PANEL_W         = PANEL_W_WIDE   # max width (used to size the window once)
PILL_H     = 38
MAX_LINES  = 6
LINE_H     = 20
TX_PAD_V   = 11
TX_PAD_H   = 22
BOTTOM     = 64
CORNER     = 19.0
FPS        = 30

_FLOAT   = 3
_BACKING = 2
_STYLE   = NSBorderlessWindowMask | NSNonactivatingPanelMask

def _c(r, g, b, a=1.0):
    return NSColor.colorWithRed_green_blue_alpha_(r, g, b, a)

BG      = _c(0.09, 0.09, 0.11, 0.96)
RED     = _c(1.00, 0.27, 0.23, 1.0)
BLUE    = _c(0.42, 0.78, 1.00, 0.85)
WHITE   = _c(0.92, 0.92, 0.94, 1.0)
DIM     = _c(0.48, 0.48, 0.52, 1.0)
DIVIDER = _c(1.00, 1.00, 1.00, 0.07)
RING    = _c(1.00, 1.00, 1.00, 0.10)
ACCENT  = _c(0.45, 0.74, 1.00, 1.0)   # soft blue recording indicator (replaces red)

PAD   = 14
IND_W = 18
TMR_W = 40   # timer only
LBL_W = 66

def _layout(w):
    """Compute control-strip x-positions for a given panel width.
    No text state label — the indicator animation conveys state — so the
    waveform spans from the indicator all the way to the timer."""
    ind_x = PAD
    tmr_x = w - PAD - TMR_W
    wav_x = ind_x + IND_W + 8
    wav_w = tmr_x - wav_x - 10
    return ind_x, wav_x, wav_w, 0, tmr_x

# Animation constants
MORPH_SPEED   = 2.5   # units/sec for state morph (0→1)
WORD_FADE_SPD = 5.0   # alpha/sec for word fade-in
RING_SPEED    = 1.2   # ripple ring expansion speed
GAP           = 9     # gap between the control pill and the transcript block


# ---------------------------------------------------------------------------
# Paragraph styles
# ---------------------------------------------------------------------------
def _para(align, wrap=False):
    s = NSMutableParagraphStyle.alloc().init()
    s.setAlignment_(align)
    s.setLineBreakMode_(NSLineBreakByWordWrapping if wrap else NSLineBreakByTruncatingTail)
    return s

_FONT_SM  = NSFont.systemFontOfSize_weight_(12.0, 0.2)
_FONT_MED = NSFont.systemFontOfSize_weight_(12.5, 0.2)
_FONT_TX  = NSFont.systemFontOfSize_weight_(13.5, 0.3)
# Italic variant for not-yet-committed ("settling") words
_FONT_TX_SETTLING = NSFont.fontWithName_size_("HelveticaNeue-Italic", 13.5) or _FONT_TX

_ATTRS_STATE = {NSFontAttributeName: _FONT_MED, NSForegroundColorAttributeName: WHITE,
                NSParagraphStyleAttributeName: _para(NSTextAlignmentLeft)}
_ATTRS_TIMER = {NSFontAttributeName: _FONT_SM,  NSForegroundColorAttributeName: DIM,
                NSParagraphStyleAttributeName: _para(NSTextAlignmentRight)}


def _draw_text(text, attrs, rect):
    if text:
        NSString.stringWithString_(text).drawInRect_withAttributes_(rect, attrs)


def _vcenter_rect(x, w, h, strip_y, strip_h):
    return NSMakeRect(x, strip_y + (strip_h - h) / 2, w, h)


# ---------------------------------------------------------------------------
# Canvas — single view draws everything
# ---------------------------------------------------------------------------
class _PillCanvas(NSView):

    def initWithFrame_(self, frame):
        self = objc.super(_PillCanvas, self).initWithFrame_(frame)
        if self is None: return None
        # Core state
        self._state        = "idle"
        self._phase        = 0.0       # animation clock (0-1 cycling)
        self._levels       = [0.02] * 52
        self._timer_str    = ""

        # State morph: 0 = full red dot, 1 = full three dots
        self._morph        = 0.0
        self._morph_target = 0.0

        # Ripple rings: list of (start_time, max_radius, color)
        self._rings        = []

        # Word-by-word transcript: [(word, alpha, settled), ...]
        self._word_alphas  = []

        return self

    def isOpaque(self): return False

    # -- setters -----------------------------------------------------------

    def setState_(self, s):
        prev = self._state
        self._state = s
        # Trigger morph
        if s == "transcribing":
            self._morph_target = 1.0
            self._rings = []   # clear rings
        elif s == "recording":
            self._morph_target = 0.0
            # Spawn a ripple ring on recording start
            self._rings = [(time.time(), 28.0, ACCENT)]
        elif s == "idle":
            self._morph_target = 0.0
            self._morph = 0.0
        self.setNeedsDisplay_(True)

    def setPhase_(self, p):
        dt = 1.0 / FPS
        self._phase = p

        # Advance morph
        diff = self._morph_target - self._morph
        if abs(diff) > 0.01:
            self._morph += math.copysign(min(abs(diff), MORPH_SPEED * dt), diff)
            self._morph = max(0.0, min(1.0, self._morph))

        # Advance word alphas
        for i, (w, a, s) in enumerate(self._word_alphas):
            if a < 1.0:
                self._word_alphas[i] = (w, min(1.0, a + WORD_FADE_SPD * dt), s)

        self.setNeedsDisplay_(True)

    def setLevels_(self, lv):
        self._levels = list(lv)
        self.setNeedsDisplay_(True)

    def setTimer_(self, t):
        self._timer_str = t or ""
        self.setNeedsDisplay_(True)

    def setWords_(self, parts):
        """parts = [committed_str, tail_str].
        Committed words render solid (settled=True); tail words shimmer (settling)."""
        committed, tail = parts
        new = [(w, True) for w in committed.split()] + \
              [(w, False) for w in tail.split()]
        old = self._word_alphas
        kept = []
        for i, (w, settled) in enumerate(new):
            if i < len(old) and old[i][0] == w:
                kept.append((w, old[i][1], settled))   # keep fade alpha, update settled
            else:
                kept.append((w, 0.0, settled))         # new word → fade in
        self._word_alphas = kept
        self.setNeedsDisplay_(True)

    def clearWords_(self, _=None):
        self._word_alphas = []
        self.setNeedsDisplay_(True)

    # -- drawing -----------------------------------------------------------

    def drawRect_(self, rect):
        total_h = rect.size.height
        w = rect.size.width

        # --- Control pill: fixed compact width, centred at the bottom ---------
        pill_w = PANEL_W_COMPACT
        pill_x = (w - pill_w) / 2
        pill_rect = NSMakeRect(pill_x, 0, pill_w, PILL_H)
        self._draw_block(pill_rect, green=(self._state == "done"))

        ind_x, wav_x, wav_w, lbl_x, tmr_x = _layout(pill_w)
        ind_x += pill_x; wav_x += pill_x; lbl_x += pill_x; tmr_x += pill_x

        self._draw_indicator(ind_x + IND_W / 2, PILL_H / 2)
        self._draw_waveform(NSMakeRect(wav_x, (PILL_H - 22) / 2, wav_w, 22))

        _draw_text(self._timer_str, _ATTRS_TIMER,
                   _vcenter_rect(tmr_x, TMR_W, 16, 0, PILL_H))

        # --- Transcript: separate floating block above the pill --------------
        if total_h > PILL_H + GAP and self._word_alphas:
            tx_rect = NSMakeRect(0, PILL_H + GAP, w, total_h - PILL_H - GAP)
            self._draw_block(tx_rect, green=False)
            self._draw_transcript(tx_rect)

    def _draw_block(self, r, green=False):
        """Draw a rounded background block with border."""
        path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(r, CORNER, CORNER)
        BG.setFill(); path.fill()
        inset = NSMakeRect(r.origin.x + 0.5, r.origin.y + 0.5,
                           r.size.width - 1, r.size.height - 1)
        border = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            inset, CORNER - 0.5, CORNER - 0.5)
        if green:
            border.setLineWidth_(1.6); _c(0.30, 0.85, 0.45, 0.85).setStroke()
        else:
            border.setLineWidth_(1.0); RING.setStroke()
        border.stroke()

    def _draw_transcript(self, r):
        """Draw words inside rect r, with confidence shimmer:
        committed words are solid white, settling words are dim + italic + pulsing."""
        if not self._word_alphas:
            return
        max_w = r.size.width - TX_PAD_H * 2

        # Lay out into lines (measuring with the committed font for stable widths)
        lines, cur, cur_w = [], [], 0.0
        for word, alpha, settled in self._word_alphas:
            sz = NSString.stringWithString_(word + " ").sizeWithAttributes_(
                {NSFontAttributeName: _FONT_TX})
            if cur_w + sz.width > max_w and cur:
                lines.append(cur); cur = [(word, alpha, settled, sz.width)]; cur_w = sz.width
            else:
                cur.append((word, alpha, settled, sz.width)); cur_w += sz.width
        if cur:
            lines.append(cur)
        lines = lines[-MAX_LINES:]

        # Settling words gently pulse in brightness
        shimmer = 0.55 + 0.18 * math.sin(self._phase * 2 * math.pi * 1.4)

        y = r.origin.y + r.size.height - TX_PAD_V - LINE_H
        for line in lines:
            x = r.origin.x + TX_PAD_H
            for word, alpha, settled, wdt in line:
                if settled:
                    col = NSColor.colorWithRed_green_blue_alpha_(0.94, 0.94, 0.96, alpha)
                else:
                    # Same font as committed (keeps box balanced) — distinguish
                    # settling words by a dim, gently pulsing blue instead of italic.
                    col = NSColor.colorWithRed_green_blue_alpha_(
                        0.70, 0.82, 1.0, alpha * shimmer)
                attrs = {NSFontAttributeName: _FONT_TX,
                         NSForegroundColorAttributeName: col,
                         NSParagraphStyleAttributeName: _para(NSTextAlignmentLeft)}
                NSString.stringWithString_(word + " ").drawAtPoint_withAttributes_(
                    NSMakePoint(x, y), attrs)
                x += wdt
            y -= LINE_H

    def _draw_sparkle(self, cx, cy):
        """A 4-pointed star (sparkle) that pulses — signals AI enhancement."""
        pulse = 0.6 + 0.4 * math.sin(self._phase * 2 * math.pi)
        outer = 9 * pulse
        inner = 2.6 * pulse
        # Gradient-ish accent colour (indigo→blue feel)
        _c(0.55, 0.80, 1.0, 0.95).setFill()
        pts = []
        for i in range(8):
            ang = math.pi / 2 + i * math.pi / 4
            r = outer if i % 2 == 0 else inner
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
        path = NSBezierPath.bezierPath()
        path.moveToPoint_(NSMakePoint(*pts[0]))
        for p in pts[1:]:
            path.lineToPoint_(NSMakePoint(*p))
        path.closePath()
        path.fill()

    def _draw_check(self, cx, cy):
        """A green checkmark — shown briefly on successful paste."""
        _c(0.30, 0.85, 0.45, 1.0).setStroke()
        path = NSBezierPath.bezierPath()
        path.setLineWidth_(2.4)
        path.moveToPoint_(NSMakePoint(cx - 6, cy))
        path.lineToPoint_(NSMakePoint(cx - 1.5, cy - 5))
        path.lineToPoint_(NSMakePoint(cx + 6.5, cy + 5))
        path.stroke()

    def _draw_indicator(self, cx, cy):
        if self._state == "polishing":
            self._draw_sparkle(cx, cy)
            return
        if self._state == "done":
            self._draw_check(cx, cy)
            return

        m = self._morph  # 0=dot, 1=three dots

        if m < 0.01:
            # Pure red dot with ripple rings
            now = time.time()
            for t0, max_r, color in self._rings:
                elapsed = (now - t0) * RING_SPEED
                if elapsed < 1.0:
                    r = max_r * elapsed
                    a = (1.0 - elapsed) * 0.3
                    _c(color.redComponent(), color.greenComponent(),
                       color.blueComponent(), a).setFill()
                    NSBezierPath.bezierPathWithOvalInRect_(
                        NSMakeRect(cx-r, cy-r, r*2, r*2)).fill()
            # Soft blue pulsing dot (replaces the red record button)
            pulse = 0.5 + 0.5 * math.sin(self._phase * 2 * math.pi)
            rr = 9 + 4 * pulse
            _c(0.45, 0.74, 1.0, 0.20 * (1 - pulse)).setFill()
            NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(cx-rr, cy-rr, rr*2, rr*2)).fill()
            ACCENT.setFill()
            NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(cx-7, cy-7, 14, 14)).fill()

        elif m > 0.99:
            # Pure three dots
            sp = 7.0; dr = 2.5
            for i in range(3):
                t = (self._phase + i / 3.0) % 1.0
                dy = -4 * math.sin(t * math.pi) if t < 1.0 else 0
                dot_cx = cx + (i - 1) * sp
                DIM.setFill()
                NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(dot_cx-dr, cy-dr+dy, dr*2, dr*2)).fill()

        else:
            # Morphing: dot shrinks toward center, dots fade in
            dot_scale = 1.0 - m
            r = 7 * dot_scale
            if r > 0.5:
                _c(0.45, 0.74, 1.0, 1.0 - m).setFill()
                NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(cx-r, cy-r, r*2, r*2)).fill()
            # Three dots fading in
            sp = 7.0; dr = 2.5
            for i in range(3):
                t = (self._phase + i / 3.0) % 1.0
                dy = -4 * math.sin(t * math.pi) * m if t < 1.0 else 0
                dot_cx = cx + (i - 1) * sp * m  # spread from center
                _c(0.48, 0.48, 0.52, m).setFill()
                NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(dot_cx-dr, cy-dr+dy, dr*2, dr*2)).fill()

    def _draw_waveform(self, rect):
        w, h = rect.size.width, rect.size.height
        cy = rect.origin.y + h / 2
        n = len(self._levels)
        if n < 2: return

        # Breathing idle animation when not recording
        if self._state != "recording":
            breath = 0.06 + 0.04 * math.sin(self._phase * 2 * math.pi * 0.4)
            base = [breath + 0.03 * math.sin(
                self._phase * 2 * math.pi * 0.7 + i * 0.4) for i in range(n)]
            active = False
        else:
            min_amp = 0.08
            base = [max(min_amp, lv) for lv in self._levels]
            active = True

        xs  = [rect.origin.x + i * w / (n - 1) for i in range(n)]
        amp = h / 2 * 0.85

        # Edge taper so lines converge to the centre at both ends
        taper = 5
        env = list(base)
        for i in range(taper):
            t = i / taper
            env[i] *= t
            env[-(i + 1)] *= t

        # Three overlapping stroked lines, each oscillating around the centre
        # with a different travelling phase — reads as moving lines, not a solid.
        if active:
            lines = [
                (_c(0.42, 0.78, 1.00, 0.95), 0.0, 1.5),
                (_c(0.40, 0.72, 1.00, 0.60), 2.1, 1.3),
                (_c(0.60, 0.66, 1.00, 0.45), 4.2, 1.1),
            ]
        else:
            lines = [
                (_c(0.52, 0.52, 0.56, 0.85), 0.0, 1.3),
                (_c(0.52, 0.52, 0.56, 0.45), 2.5, 1.1),
            ]

        speed = 1.3 if active else 0.5
        for color, shift, lw in lines:
            pts = []
            for i in range(n):
                osc = math.sin(self._phase * 2 * math.pi * speed + i * 0.5 + shift)
                pts.append((xs[i], cy + amp * env[i] * osc))
            path = _catmull(pts)   # open smooth curve (no fill)
            color.setStroke()
            path.setLineWidth_(lw)
            path.stroke()


def _catmull(pts):
    path = NSBezierPath.bezierPath(); n = len(pts)
    path.moveToPoint_(NSMakePoint(*pts[0]))
    for i in range(1, n):
        p0=pts[max(0,i-2)]; p1=pts[i-1]; p2=pts[i]; p3=pts[min(n-1,i+1)]
        cp1=(p1[0]+(p2[0]-p0[0])/6, p1[1]+(p2[1]-p0[1])/6)
        cp2=(p2[0]-(p3[0]-p1[0])/6, p2[1]-(p3[1]-p1[1])/6)
        path.curveToPoint_controlPoint1_controlPoint2_(
            NSMakePoint(*p2), NSMakePoint(*cp1), NSMakePoint(*cp2))
    return path

def _catmull_ext(path, pts):
    n = len(pts)
    for i in range(1, n):
        p0=pts[max(0,i-2)]; p1=pts[i-1]; p2=pts[i]; p3=pts[min(n-1,i+1)]
        cp1=(p1[0]+(p2[0]-p0[0])/6, p1[1]+(p2[1]-p0[1])/6)
        cp2=(p2[0]-(p3[0]-p1[0])/6, p2[1]-(p3[1]-p1[1])/6)
        path.curveToPoint_controlPoint1_controlPoint2_(
            NSMakePoint(*p2), NSMakePoint(*cp1), NSMakePoint(*cp2))


# ---------------------------------------------------------------------------
# Overlay panel
# ---------------------------------------------------------------------------
class OverlayPanel(NSObject):

    def init(self):
        self = objc.super(OverlayPanel, self).init()
        if self is None: return None
        self._panel        = None
        self._canvas       = None
        self._anim_timer   = None
        self._anim_phase   = 0.0
        self._record_start = 0.0
        self._state        = "idle"
        self._transcript   = ""
        self._committed_disp = ""
        self._tail_disp      = ""
        self._power        = ""
        return self

    def _setup(self):
        sf = NSScreen.mainScreen().frame()
        x = (sf.size.width - PANEL_W) / 2 + sf.origin.x
        y = sf.origin.y + BOTTOM
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, PANEL_W, PILL_H), _STYLE, _BACKING, False)
        panel.setLevel_(_FLOAT + 1)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(True)
        panel.setAlphaValue_(0.0)
        canvas = _PillCanvas.alloc().initWithFrame_(NSMakeRect(0, 0, PANEL_W, PILL_H))
        canvas.setWantsLayer_(True)
        canvas.layer().setBackgroundColor_(NSColor.clearColor().CGColor())
        panel.setContentView_(canvas)
        self._panel  = panel
        self._canvas = canvas

    def _updateLayout(self):
        if not self._panel: return
        combined = (self._committed_disp + " " + self._tail_disp).strip()
        words = combined.split() if combined else []

        if not words:
            panel_w = PANEL_W_COMPACT
            text_h = 0
        else:
            attrs = {NSFontAttributeName: _FONT_TX}
            word_w = [NSString.stringWithString_(w + " ").sizeWithAttributes_(attrs).width
                      for w in words]
            single_line = sum(word_w)
            # Grow width with content: from compact up to the wide maximum
            panel_w = max(PANEL_W_COMPACT,
                          min(PANEL_W_WIDE, single_line + TX_PAD_H * 2))
            max_w = panel_w - TX_PAD_H * 2
            lines = 1; cur_w = 0.0
            for wd in word_w:
                if cur_w + wd > max_w and cur_w > 0:
                    lines += 1; cur_w = wd
                else:
                    cur_w += wd
            text_h = min(lines, MAX_LINES) * LINE_H + TX_PAD_V * 2

        # Two-block layout: pill at bottom, gap, transcript block above
        total_h = PILL_H + (GAP + text_h if text_h else 0)
        sf = NSScreen.mainScreen().frame()
        x = (sf.size.width - panel_w) / 2 + sf.origin.x
        y = sf.origin.y + BOTTOM
        new_frame = NSMakeRect(x, y, panel_w, total_h)

        NSAnimationContext.beginGrouping()
        NSAnimationContext.currentContext().setDuration_(0.18)
        self._panel.animator().setFrame_display_(new_frame, True)
        NSAnimationContext.endGrouping()

        self._canvas.setFrame_(NSMakeRect(0, 0, panel_w, total_h))
        self._canvas.setWords_([self._committed_disp, self._tail_disp])

    def _startTimer(self):
        if self._anim_timer: return
        self._anim_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0/FPS, self, "animTick:", None, True)

    def _stopTimer(self):
        if self._anim_timer:
            self._anim_timer.invalidate(); self._anim_timer = None

    def animTick_(self, _):
        self._anim_phase = (self._anim_phase + 1.0/FPS) % 1.0
        if not self._canvas:
            return
        self._canvas.setPhase_(self._anim_phase)
        # Right-side text: just the recording timer (CPU% lives in the menu)
        if self._state == "recording":
            e = time.time() - self._record_start
            self._canvas.setTimer_(f"{int(e//60)}:{int(e%60):02d}")

    # -- main thread selectors ------------------------------------------

    def show(self):
        if self._panel is None: self._setup()
        self._transcript = ""
        self._committed_disp = ""
        self._tail_disp = ""
        self._canvas.clearWords_(None)
        self._updateLayout()
        self._panel.setAlphaValue_(1.0)
        self._panel.orderFrontRegardless()
        self._startTimer()

    def hide(self):
        self._stopTimer()
        if self._panel:
            self._panel.setAlphaValue_(0.0)
            self._panel.orderOut_(None)

    def setStateObj_(self, state):
        self._state = state
        if self._canvas: self._canvas.setState_(state)
        if state == "recording":
            self._record_start = time.time()
            if self._canvas: self._canvas.setTimer_("0:00")
        elif state == "idle":
            if self._canvas:
                self._canvas.setTimer_("")
                self._canvas.clearWords_(None)
            self._transcript = ""
            self._committed_disp = ""
            self._tail_disp = ""
            self._updateLayout()

    def setLevelsObj_(self, lvl):
        if self._canvas: self._canvas.setLevels_(lvl)

    def setTextObj_(self, text):
        # Single-string update → all committed (used for final text)
        text = text or ""
        if text == self._committed_disp and not self._tail_disp:
            return
        self._committed_disp = text
        self._tail_disp = ""
        self._updateLayout()

    def setTextPartsObj_(self, parts):
        committed, tail = parts
        if committed == self._committed_disp and tail == self._tail_disp:
            return
        self._committed_disp = committed or ""
        self._tail_disp = tail or ""
        self._updateLayout()

    def setPowerObj_(self, p):
        self._power = p or ""

    # -- thread-safe wrappers ------------------------------------------

    def push_power(self, p):
        self.performSelectorOnMainThread_withObject_waitUntilDone_("setPowerObj:", p, False)
    def push_state(self, s):
        self.performSelectorOnMainThread_withObject_waitUntilDone_("setStateObj:", s, False)
    def push_levels(self, lvl):
        self.performSelectorOnMainThread_withObject_waitUntilDone_("setLevelsObj:", list(lvl), False)
    def push_text(self, text):
        self.performSelectorOnMainThread_withObject_waitUntilDone_("setTextObj:", text, False)
    def push_text_parts(self, committed, tail):
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "setTextPartsObj:", [committed, tail], False)
    def show_async(self):
        self.performSelectorOnMainThread_withObject_waitUntilDone_("show", None, False)
    def hide_async(self):
        self.performSelectorOnMainThread_withObject_waitUntilDone_("hide", None, False)
