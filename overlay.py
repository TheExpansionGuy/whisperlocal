"""Pill overlay — grows upward as transcript builds, truly transparent window."""
import math
import time
import objc
from AppKit import (
    NSBorderlessWindowMask, NSBezierPath, NSColor, NSFont,
    NSMakeRect, NSNonactivatingPanelMask, NSPanel, NSScreen,
    NSTextField, NSTextView, NSScrollView,
    NSTextAlignmentCenter, NSTextAlignmentLeft, NSTextAlignmentRight,
    NSView, NSFocusRingTypeNone, NSViewWidthSizable, NSViewHeightSizable,
)
from Foundation import NSMakePoint, NSMakeSize, NSObject, NSTimer, NSMutableAttributedString, NSAttributedString

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PANEL_W   = 600
PILL_H    = 52       # fixed bottom strip height
MAX_LINES = 6
LINE_H    = 20
TEXT_PAD_V = 10
TEXT_PAD_H = 20
BOTTOM    = 64
CORNER    = 26.0
FPS       = 30

_FLOAT   = 3
_BACKING = 2
_STYLE   = NSBorderlessWindowMask | NSNonactivatingPanelMask

def _c(r, g, b, a=1.0):
    return NSColor.colorWithRed_green_blue_alpha_(r, g, b, a)

BG      = _c(0.11, 0.11, 0.12, 0.97)
RED     = _c(1.00, 0.27, 0.23, 1.0)
BLUE    = _c(0.42, 0.78, 1.00, 0.82)
WHITE   = _c(0.90, 0.90, 0.92, 1.0)
DIM     = _c(0.50, 0.50, 0.52, 1.0)
DIVIDER = _c(1.00, 1.00, 1.00, 0.08)
RING    = _c(1.00, 1.00, 1.00, 0.09)

# Pill row layout
PAD    = 18
IND_W  = 22
IND_X  = PAD
TMR_W  = 46
TMR_X  = PANEL_W - PAD - TMR_W
LBL_W  = 82
LBL_X  = TMR_X - LBL_W - 6
WAV_X  = IND_X + IND_W + 12
WAV_W  = LBL_X - WAV_X - 8


# ---------------------------------------------------------------------------
# Indicator view
# ---------------------------------------------------------------------------
class _IndicatorView(NSView):
    def initWithFrame_(self, frame):
        self = objc.super(_IndicatorView, self).initWithFrame_(frame)
        if self is None: return None
        self._state = "idle"
        self._phase = 0.0
        self.setWantsLayer_(False)
        return self

    def isOpaque(self): return False

    def setState_(self, s):
        self._state = s
        self.setNeedsDisplay_(True)

    def setPhase_(self, p):
        self._phase = p
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        cx = rect.size.width / 2
        cy = rect.size.height / 2
        if self._state == "recording":
            pulse = 0.5 + 0.5 * math.sin(self._phase * 2 * math.pi)
            rr = 9 + 5 * pulse
            _c(1.0, 0.27, 0.23, 0.20 * (1 - pulse)).setFill()
            NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(cx - rr, cy - rr, rr * 2, rr * 2)).fill()
            RED.setFill()
            NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(cx - 7, cy - 7, 14, 14)).fill()
        elif self._state == "transcribing":
            spacing = 9.0; dot_r = 3.0; x0 = cx - spacing
            for i in range(3):
                t = (self._phase + i / 3.0) % 1.0
                dy = -5 * math.sin(t * math.pi) if t < 1.0 else 0
                DIM.setFill()
                NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(x0 + i * spacing - dot_r,
                               cy - dot_r + dy, dot_r * 2, dot_r * 2)).fill()


# ---------------------------------------------------------------------------
# Waveform view
# ---------------------------------------------------------------------------
class _WaveformView(NSView):
    def initWithFrame_(self, frame):
        self = objc.super(_WaveformView, self).initWithFrame_(frame)
        if self is None: return None
        self._levels = [0.02] * 52
        self._active = False
        self.setWantsLayer_(False)
        return self

    def isOpaque(self): return False

    def setLevels_(self, lvl):
        self._levels = list(lvl)
        self.setNeedsDisplay_(True)

    def setActive_(self, a):
        self._active = a
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        w, h = rect.size.width, rect.size.height
        cy = h / 2
        n = len(self._levels)
        if n < 2: return
        xs  = [i * w / (n - 1) for i in range(n)]
        amp = cy * 0.84
        top = [(xs[i], cy - self._levels[i] * amp) for i in range(n)]
        bot = [(xs[i], cy + self._levels[i] * amp) for i in range(n - 1, -1, -1)]
        (BLUE if self._active else DIM).setFill()
        p = _smooth_path(top)
        _smooth_extend(p, bot)
        p.closePath(); p.fill()


def _smooth_path(pts):
    path = NSBezierPath.bezierPath()
    n = len(pts)
    path.moveToPoint_(NSMakePoint(*pts[0]))
    for i in range(1, n):
        p0=pts[max(0,i-2)]; p1=pts[i-1]; p2=pts[i]; p3=pts[min(n-1,i+1)]
        cp1=(p1[0]+(p2[0]-p0[0])/6, p1[1]+(p2[1]-p0[1])/6)
        cp2=(p2[0]-(p3[0]-p1[0])/6, p2[1]-(p3[1]-p1[1])/6)
        path.curveToPoint_controlPoint1_controlPoint2_(
            NSMakePoint(*p2), NSMakePoint(*cp1), NSMakePoint(*cp2))
    return path

def _smooth_extend(path, pts):
    n = len(pts)
    for i in range(1, n):
        p0=pts[max(0,i-2)]; p1=pts[i-1]; p2=pts[i]; p3=pts[min(n-1,i+1)]
        cp1=(p1[0]+(p2[0]-p0[0])/6, p1[1]+(p2[1]-p0[1])/6)
        cp2=(p2[0]-(p3[0]-p1[0])/6, p2[1]-(p3[1]-p1[1])/6)
        path.curveToPoint_controlPoint1_controlPoint2_(
            NSMakePoint(*p2), NSMakePoint(*cp1), NSMakePoint(*cp2))


# ---------------------------------------------------------------------------
# Root view — draws the pill shape for the entire panel
# ---------------------------------------------------------------------------
class _PillView(NSView):
    def isOpaque(self): return False

    def drawRect_(self, rect):
        # Full rounded pill
        pill = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            rect, CORNER, CORNER)
        BG.setFill(); pill.fill()
        # Border
        inset = NSMakeRect(0.5, 0.5, rect.size.width-1, rect.size.height-1)
        border = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            inset, CORNER-0.5, CORNER-0.5)
        border.setLineWidth_(1.0); RING.setStroke(); border.stroke()
        # Divider above pill strip (only when taller than PILL_H)
        if rect.size.height > PILL_H + 2:
            div = NSBezierPath.bezierPath()
            div.moveToPoint_(NSMakePoint(PAD, PILL_H))
            div.lineToPoint_(NSMakePoint(rect.size.width - PAD, PILL_H))
            div.setLineWidth_(0.5); DIVIDER.setStroke(); div.stroke()


# ---------------------------------------------------------------------------
# Overlay panel
# ---------------------------------------------------------------------------
class OverlayPanel(NSObject):

    def init(self):
        self = objc.super(OverlayPanel, self).init()
        if self is None: return None
        self._panel        = None
        self._root         = None
        self._indicator    = None
        self._waveform     = None
        self._lbl_state    = None
        self._lbl_timer    = None
        self._textview     = None
        self._anim_timer   = None
        self._anim_phase   = 0.0
        self._record_start = 0.0
        self._state        = "idle"
        self._transcript   = ""
        return self

    # ------------------------------------------------------------------
    def _setup(self):
        sf = NSScreen.mainScreen().frame()
        x = (sf.size.width - PANEL_W) / 2 + sf.origin.x
        y = sf.origin.y + BOTTOM

        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, PANEL_W, PILL_H), _STYLE, _BACKING, False)
        panel.setLevel_(_FLOAT + 1)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())   # ← key: truly transparent
        panel.setHasShadow_(True)
        panel.setAlphaValue_(0.0)

        root = _PillView.alloc().initWithFrame_(NSMakeRect(0, 0, PANEL_W, PILL_H))
        panel.setContentView_(root)
        self._root = root

        cy = PILL_H / 2

        # Indicator
        ind = _IndicatorView.alloc().initWithFrame_(
            NSMakeRect(IND_X, (PILL_H - IND_W) / 2, IND_W, IND_W))
        root.addSubview_(ind)
        self._indicator = ind

        # Waveform
        wf_h = 30
        wf = _WaveformView.alloc().initWithFrame_(
            NSMakeRect(WAV_X, (PILL_H - wf_h) / 2, WAV_W, wf_h))
        root.addSubview_(wf)
        self._waveform = wf

        # State label
        sl = _lbl(LBL_X, (PILL_H-20)/2, LBL_W, 20, "", WHITE, 12.5)
        root.addSubview_(sl)
        self._lbl_state = sl

        # Timer label
        tl = _lbl(TMR_X, (PILL_H-20)/2, TMR_W, 20, "", DIM, 12.0, right=True)
        root.addSubview_(tl)
        self._lbl_timer = tl

        # NSTextView for transcript (hidden until text arrives)
        tv = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, 0, 0))
        tv.setEditable_(False)
        tv.setSelectable_(False)
        tv.setDrawsBackground_(False)
        tv.setBackgroundColor_(NSColor.clearColor())
        tv.setTextColor_(WHITE)
        tv.setFont_(NSFont.systemFontOfSize_weight_(13.0, 0.2))
        tv.textContainer().setLineFragmentPadding_(0)
        tv.setAlignment_(NSTextAlignmentCenter)
        tv.setWantsLayer_(False)
        root.addSubview_(tv)
        self._textview = tv

        self._panel = panel

    # ------------------------------------------------------------------
    def _updateLayout(self):
        """Resize panel upward to fit current transcript text."""
        if not self._panel: return

        text = self._transcript.strip()
        if not text:
            text_h = 0
        else:
            # Estimate lines needed
            chars_per_line = int((PANEL_W - TEXT_PAD_H * 2) / 7.5)
            words = text.split()
            lines, cur = 1, 0
            for w in words:
                if cur + len(w) + 1 > chars_per_line:
                    lines += 1; cur = len(w)
                else:
                    cur += len(w) + 1
            lines = min(lines, MAX_LINES)
            text_h = lines * LINE_H + TEXT_PAD_V * 2

        total_h = PILL_H + text_h

        # Keep bottom edge fixed — move y up as panel grows
        sf = NSScreen.mainScreen().frame()
        x = (sf.size.width - PANEL_W) / 2 + sf.origin.x
        y = sf.origin.y + BOTTOM

        self._panel.setFrame_display_(
            NSMakeRect(x, y, PANEL_W, total_h), True)
        self._root.setFrame_(NSMakeRect(0, 0, PANEL_W, total_h))

        # Reposition pill-row subviews (always at bottom PILL_H strip)
        for view in [self._indicator, self._waveform, self._lbl_state, self._lbl_timer]:
            if view:
                f = view.frame()
                view.setFrame_(NSMakeRect(f.origin.x, f.origin.y, f.size.width, f.size.height))

        # Resize transcript view
        if text_h > 0:
            self._textview.setFrame_(
                NSMakeRect(TEXT_PAD_H, PILL_H + TEXT_PAD_V,
                           PANEL_W - TEXT_PAD_H * 2, text_h - TEXT_PAD_V * 2))
            self._textview.setString_(text)
        else:
            self._textview.setFrame_(NSMakeRect(0, 0, 0, 0))
            self._textview.setString_("")

        self._root.setNeedsDisplay_(True)

    # ------------------------------------------------------------------
    def _startTimer(self):
        if self._anim_timer: return
        self._anim_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / FPS, self, "animTick:", None, True)

    def _stopTimer(self):
        if self._anim_timer:
            self._anim_timer.invalidate()
            self._anim_timer = None

    def animTick_(self, _):
        self._anim_phase = (self._anim_phase + 1.0 / FPS) % 1.0
        if self._indicator: self._indicator.setPhase_(self._anim_phase)
        if self._state == "recording" and self._lbl_timer:
            e = time.time() - self._record_start
            self._lbl_timer.setStringValue_(f"{int(e//60)}:{int(e%60):02d}")

    # ------------------------------------------------------------------
    def show(self):
        if self._panel is None: self._setup()
        self._transcript = ""
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
        if self._indicator: self._indicator.setState_(state)
        if self._waveform:  self._waveform.setActive_(state == "recording")
        if self._lbl_state:
            self._lbl_state.setStringValue_(
                {"recording": "Listening", "transcribing": "Transcribing"}.get(state, ""))
        if self._lbl_timer:
            if state == "recording":
                self._record_start = time.time()
                self._lbl_timer.setStringValue_("0:00")
            elif state == "idle":
                self._lbl_timer.setStringValue_("")
        if state == "idle":
            self._transcript = ""
            self._updateLayout()

    def setLevelsObj_(self, lvl):
        if self._waveform: self._waveform.setLevels_(lvl)

    def setTextObj_(self, text):
        self._transcript = text or ""
        self._updateLayout()

    # ------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Label helper — no borders, no background, no focus ring
# ---------------------------------------------------------------------------
def _lbl(x, y, w, h, text, color, size, right=False):
    l = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, w, h))
    l.setEditable_(False); l.setBordered_(False)
    l.setDrawsBackground_(False)
    l.setBackgroundColor_(NSColor.clearColor())
    l.cell().setDrawsBackground_(False)
    l.cell().setBackgroundColor_(NSColor.clearColor())
    l.setFocusRingType_(NSFocusRingTypeNone)
    l.setTextColor_(color)
    l.setFont_(NSFont.systemFontOfSize_weight_(size, 0.3))
    l.setAlignment_(NSTextAlignmentRight if right else NSTextAlignmentLeft)
    l.setStringValue_(text)
    return l
