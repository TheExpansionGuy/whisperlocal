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
PANEL_W    = 600
PILL_H     = 52
MAX_LINES  = 6
LINE_H     = 20
TX_PAD_V   = 12
TX_PAD_H   = 24
BOTTOM     = 64
CORNER     = 26.0
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

PAD   = 18
IND_W = 22;  IND_X = PAD
TMR_W = 46;  TMR_X = PANEL_W - PAD - TMR_W
LBL_W = 84;  LBL_X = TMR_X - LBL_W - 6
WAV_X = IND_X + IND_W + 12
WAV_W = LBL_X - WAV_X - 8

# Animation constants
MORPH_SPEED   = 2.5   # units/sec for state morph (0→1)
WORD_FADE_SPD = 5.0   # alpha/sec for word fade-in
RING_SPEED    = 1.2   # ripple ring expansion speed


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

        # Word-by-word transcript: list of (word, alpha)
        self._word_alphas  = []        # [(word_str, alpha_float), ...]
        self._last_words   = []        # words already fully revealed

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
            self._rings = [(time.time(), 28.0, RED)]
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
        for i, (w, a) in enumerate(self._word_alphas):
            if a < 1.0:
                self._word_alphas[i] = (w, min(1.0, a + WORD_FADE_SPD * dt))

        self.setNeedsDisplay_(True)

    def setLevels_(self, lv):
        self._levels = list(lv)
        self.setNeedsDisplay_(True)

    def setTimer_(self, t):
        self._timer_str = t or ""
        self.setNeedsDisplay_(True)

    def setWords_(self, new_words):
        """Diff against existing words; fade in only the new ones."""
        old = [w for w, _ in self._word_alphas]
        # Find common prefix
        common = 0
        for i, (ow, nw) in enumerate(zip(old, new_words)):
            if ow == nw: common = i + 1
            else: break

        # Keep already-revealed words, add new ones at alpha=0
        kept = self._word_alphas[:common]
        for w in new_words[common:]:
            kept.append((w, 0.0))
        self._word_alphas = kept
        self.setNeedsDisplay_(True)

    def clearWords_(self, _=None):
        self._word_alphas = []
        self.setNeedsDisplay_(True)

    # -- drawing -----------------------------------------------------------

    def drawRect_(self, rect):
        total_h = rect.size.height

        pill = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            rect, CORNER, CORNER)
        pill.addClip()

        # Background
        BG.setFill(); pill.fill()

        # Border ring
        inset = NSMakeRect(0.5, 0.5, rect.size.width - 1, rect.size.height - 1)
        border = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            inset, CORNER - 0.5, CORNER - 0.5)
        border.setLineWidth_(1.0); RING.setStroke(); border.stroke()

        # Divider + transcript
        if total_h > PILL_H + 4:
            div = NSBezierPath.bezierPath()
            div.moveToPoint_(NSMakePoint(PAD, PILL_H))
            div.lineToPoint_(NSMakePoint(rect.size.width - PAD, PILL_H))
            div.setLineWidth_(0.5); DIVIDER.setStroke(); div.stroke()
            self._draw_transcript(total_h)

        # Pill strip
        self._draw_indicator(IND_X + IND_W / 2, PILL_H / 2)
        self._draw_waveform(NSMakeRect(WAV_X, (PILL_H - 30) / 2, WAV_W, 30))

        # Labels
        state_label = {"recording": "Listening",
                       "transcribing": "Transcribing"}.get(self._state, "")
        _draw_text(state_label, _ATTRS_STATE,
                   _vcenter_rect(LBL_X, LBL_W, 16, 0, PILL_H))
        _draw_text(self._timer_str, _ATTRS_TIMER,
                   _vcenter_rect(TMR_X, TMR_W, 16, 0, PILL_H))

    def _draw_transcript(self, total_h):
        """Draw words top-to-bottom with per-word alpha fade-in."""
        if not self._word_alphas:
            return

        max_w = PANEL_W - TX_PAD_H * 2

        # First pass: lay out words into lines
        lines = []      # list of [(word, alpha), ...]
        current_line = []
        current_w = 0.0

        for word, alpha in self._word_alphas:
            attrs = {
                NSFontAttributeName: _FONT_TX,
                NSForegroundColorAttributeName: WHITE,
                NSParagraphStyleAttributeName: _para(NSTextAlignmentLeft),
            }
            s = NSString.stringWithString_(word + " ")
            sz = s.sizeWithAttributes_(attrs)
            if current_w + sz.width > max_w and current_line:
                lines.append(current_line)
                current_line = [(word, alpha, sz.width)]
                current_w = sz.width
            else:
                current_line.append((word, alpha, sz.width))
                current_w += sz.width
        if current_line:
            lines.append(current_line)

        # Cap at MAX_LINES, showing the most recent lines
        lines = lines[-MAX_LINES:]

        # Second pass: draw top-to-bottom
        # In AppKit y-up coords: top of transcript = total_h - TX_PAD_V
        # Each line goes down by LINE_H
        y = total_h - TX_PAD_V - LINE_H
        for line in lines:
            x = TX_PAD_H
            for word, alpha, w in line:
                attrs = {
                    NSFontAttributeName: _FONT_TX,
                    NSForegroundColorAttributeName: NSColor.colorWithRed_green_blue_alpha_(
                        0.92, 0.92, 0.94, alpha),
                    NSParagraphStyleAttributeName: _para(NSTextAlignmentLeft),
                }
                s = NSString.stringWithString_(word + " ")
                s.drawAtPoint_withAttributes_(NSMakePoint(x, y), attrs)
                x += w
            y -= LINE_H

    def _draw_indicator(self, cx, cy):
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
            # Pulse
            pulse = 0.5 + 0.5 * math.sin(self._phase * 2 * math.pi)
            rr = 9 + 4 * pulse
            _c(1.0, 0.27, 0.23, 0.18 * (1 - pulse)).setFill()
            NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(cx-rr, cy-rr, rr*2, rr*2)).fill()
            RED.setFill()
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
            # Red dot shrinking
            dot_scale = 1.0 - m
            r = 7 * dot_scale
            red_a = 1.0 - m
            if r > 0.5:
                _c(1.0, 0.27, 0.23, red_a).setFill()
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
            levels = [breath + 0.03 * math.sin(
                self._phase * 2 * math.pi * 0.7 + i * 0.4) for i in range(n)]
        else:
            min_amp = 0.08
            levels = [max(min_amp, lv) for lv in self._levels]

        taper = 5
        for i in range(taper):
            t = i / taper
            levels[i] *= t
            levels[-(i+1)] *= t

        xs  = [rect.origin.x + i * w / (n - 1) for i in range(n)]
        amp = h / 2 * 0.84
        top = [(xs[i], cy - levels[i] * amp) for i in range(n)]
        bot = [(xs[i], cy + levels[i] * amp) for i in range(n-1, -1, -1)]

        color = BLUE if self._state == "recording" else DIM
        color.setFill()
        p = _catmull(top); _catmull_ext(p, bot); p.closePath(); p.fill()


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
        words = self._transcript.strip().split() if self._transcript.strip() else []

        if not words:
            text_h = 0
        else:
            chars = int((PANEL_W - TX_PAD_H * 2) / 7.8)
            lines = 1; cur = 0
            for w in words:
                if cur + len(w) + 1 > chars: lines += 1; cur = len(w)
                else: cur += len(w) + 1
            text_h = min(lines, MAX_LINES) * LINE_H + TX_PAD_V * 2

        total_h = PILL_H + text_h
        sf = NSScreen.mainScreen().frame()
        x = (sf.size.width - PANEL_W) / 2 + sf.origin.x
        y = sf.origin.y + BOTTOM
        new_frame = NSMakeRect(x, y, PANEL_W, total_h)

        # Animated resize
        NSAnimationContext.beginGrouping()
        NSAnimationContext.currentContext().setDuration_(0.25)
        self._panel.animator().setFrame_display_(new_frame, True)
        NSAnimationContext.endGrouping()

        self._canvas.setFrame_(NSMakeRect(0, 0, PANEL_W, total_h))
        if self._canvas:
            self._canvas.setWords_(words)

    def _startTimer(self):
        if self._anim_timer: return
        self._anim_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0/FPS, self, "animTick:", None, True)

    def _stopTimer(self):
        if self._anim_timer:
            self._anim_timer.invalidate(); self._anim_timer = None

    def animTick_(self, _):
        self._anim_phase = (self._anim_phase + 1.0/FPS) % 1.0
        if self._canvas: self._canvas.setPhase_(self._anim_phase)
        if self._state == "recording" and self._canvas:
            e = time.time() - self._record_start
            self._canvas.setTimer_(f"{int(e//60)}:{int(e%60):02d}")

    # -- main thread selectors ------------------------------------------

    def show(self):
        if self._panel is None: self._setup()
        self._transcript = ""
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
            self._updateLayout()

    def setLevelsObj_(self, lvl):
        if self._canvas: self._canvas.setLevels_(lvl)

    def setTextObj_(self, text):
        text = text or ""
        if text == self._transcript:
            return  # no change — skip relayout/animation
        self._transcript = text
        self._updateLayout()

    # -- thread-safe wrappers ------------------------------------------

    def push_state(self, s):
        self.performSelectorOnMainThread_withObject_waitUntilDone_("setStateObj:", s, False)
    def push_levels(self, lvl):
        self.performSelectorOnMainThread_withObject_waitUntilDone_("setLevelsObj:", list(lvl), False)
    def push_text(self, text):
        self.performSelectorOnMainThread_withObject_waitUntilDone_("setTextObj:", text, False)
    def show_async(self):
        self.performSelectorOnMainThread_withObject_waitUntilDone_("show", None, False)
    def hide_async(self):
        self.performSelectorOnMainThread_withObject_waitUntilDone_("hide", None, False)
