"""Pill overlay — single NSView draws everything. No subviews, no bezels possible."""
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
    NSGraphicsContext,
)
from Foundation import NSMakePoint, NSObject, NSTimer, NSString

# ---------------------------------------------------------------------------
# Layout & colours
# ---------------------------------------------------------------------------
PANEL_W    = 600
PILL_H     = 52
MAX_LINES  = 6
LINE_H     = 20
TX_PAD_V   = 10
TX_PAD_H   = 22
BOTTOM     = 64
CORNER     = 26.0
FPS        = 30

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
DIVIDER = _c(1.00, 1.00, 1.00, 0.07)
RING    = _c(1.00, 1.00, 1.00, 0.09)

# Pill-strip sub-positions
PAD   = 18
IND_W = 22;  IND_X = PAD
TMR_W = 46;  TMR_X = PANEL_W - PAD - TMR_W
LBL_W = 84;  LBL_X = TMR_X - LBL_W - 6
WAV_X = IND_X + IND_W + 12
WAV_W = LBL_X - WAV_X - 8


# ---------------------------------------------------------------------------
# Fonts & paragraph styles (built once)
# ---------------------------------------------------------------------------
def _para(align, wrap=False):
    s = NSMutableParagraphStyle.alloc().init()
    s.setAlignment_(align)
    s.setLineBreakMode_(NSLineBreakByWordWrapping if wrap else NSLineBreakByTruncatingTail)
    return s

_FONT_SM  = NSFont.systemFontOfSize_weight_(12.0, 0.2)
_FONT_MED = NSFont.systemFontOfSize_weight_(12.5, 0.2)
_FONT_TX  = NSFont.systemFontOfSize_weight_(13.0, 0.2)

_ATTRS_STATE = {NSFontAttributeName: _FONT_MED, NSForegroundColorAttributeName: WHITE,
                NSParagraphStyleAttributeName: _para(NSTextAlignmentLeft)}
_ATTRS_TIMER = {NSFontAttributeName: _FONT_SM,  NSForegroundColorAttributeName: DIM,
                NSParagraphStyleAttributeName: _para(NSTextAlignmentRight)}
_ATTRS_TX    = {NSFontAttributeName: _FONT_TX,  NSForegroundColorAttributeName: WHITE,
                NSParagraphStyleAttributeName: _para(NSTextAlignmentCenter, wrap=True)}


def _draw_text(text, attrs, rect):
    if text:
        NSString.stringWithString_(text).drawInRect_withAttributes_(rect, attrs)


def _vcenter_rect(x, w, label_h, strip_y, strip_h):
    """Return a rect vertically centred within the strip (AppKit y-up)."""
    return NSMakeRect(x, strip_y + (strip_h - label_h) / 2, w, label_h)


# ---------------------------------------------------------------------------
# Single-view that draws the entire overlay
# ---------------------------------------------------------------------------
class _PillCanvas(NSView):

    def initWithFrame_(self, frame):
        self = objc.super(_PillCanvas, self).initWithFrame_(frame)
        if self is None: return None
        self._state   = "idle"
        self._phase   = 0.0
        self._levels  = [0.02] * 52
        self._text    = ""
        self._timer   = ""
        return self

    def isOpaque(self): return False

    # -- setters called by OverlayPanel ---------------------------------
    def setState_(self, s):   self._state=s;  self.setNeedsDisplay_(True)
    def setPhase_(self, p):   self._phase=p;  self.setNeedsDisplay_(True)
    def setLevels_(self, lv): self._levels=list(lv); self.setNeedsDisplay_(True)
    def setText_(self, t):    self._text=t or "";    self.setNeedsDisplay_(True)
    def setTimer_(self, t):   self._timer=t or "";   self.setNeedsDisplay_(True)

    # -------------------------------------------------------------------
    def drawRect_(self, rect):
        total_h = rect.size.height

        # 1. Clip entire drawing to pill shape — no subview can escape this
        pill = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            rect, CORNER, CORNER)
        NSGraphicsContext.currentContext().saveGraphicsState()
        pill.addClip()

        # 2. Background fill
        BG.setFill()
        pill.fill()

        # 3. Divider between transcript and pill strip
        if total_h > PILL_H + 2:
            div = NSBezierPath.bezierPath()
            div.moveToPoint_(NSMakePoint(PAD, PILL_H))
            div.lineToPoint_(NSMakePoint(total_h - PAD, PILL_H))
            div.setLineWidth_(0.5); DIVIDER.setStroke(); div.stroke()

        # 4. Transcript text
        if self._text and total_h > PILL_H + 2:
            tx_rect = NSMakeRect(TX_PAD_H, PILL_H + TX_PAD_V,
                                 PANEL_W - TX_PAD_H * 2,
                                 total_h - PILL_H - TX_PAD_V * 2)
            _draw_text(self._text, _ATTRS_TX, tx_rect)

        # 5. Indicator (dot or dots) — centred in pill strip
        cx = IND_X + IND_W / 2
        cy = PILL_H / 2
        self._draw_indicator(cx, cy)

        # 6. Waveform
        self._draw_waveform(
            NSMakeRect(WAV_X, (PILL_H - 30) / 2, WAV_W, 30))

        # 7. State label
        _draw_text(
            {"recording": "Listening", "transcribing": "Transcribing"}.get(self._state, ""),
            _ATTRS_STATE,
            _vcenter_rect(LBL_X, LBL_W, 16, 0, PILL_H))

        # 8. Timer
        _draw_text(self._timer, _ATTRS_TIMER,
                   _vcenter_rect(TMR_X, TMR_W, 16, 0, PILL_H))

        # 9. Border ring (drawn last, on top)
        NSGraphicsContext.currentContext().restoreGraphicsState()
        inset = NSMakeRect(0.5, 0.5, rect.size.width - 1, rect.size.height - 1)
        border = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            inset, CORNER - 0.5, CORNER - 0.5)
        border.setLineWidth_(1.0); RING.setStroke(); border.stroke()

    def _draw_indicator(self, cx, cy):
        if self._state == "recording":
            pulse = 0.5 + 0.5 * math.sin(self._phase * 2 * math.pi)
            rr = 9 + 5 * pulse
            _c(1.0, 0.27, 0.23, 0.20 * (1 - pulse)).setFill()
            NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(cx-rr, cy-rr, rr*2, rr*2)).fill()
            RED.setFill()
            NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(cx-7, cy-7, 14, 14)).fill()
        elif self._state == "transcribing":
            sp=9.0; dr=3.0; x0=cx-sp
            for i in range(3):
                t = (self._phase + i / 3.0) % 1.0
                dy = -5 * math.sin(t * math.pi) if t < 1.0 else 0
                DIM.setFill()
                NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(x0+i*sp-dr, cy-dr+dy, dr*2, dr*2)).fill()

    def _draw_waveform(self, rect):
        w, h = rect.size.width, rect.size.height
        cy = rect.origin.y + h / 2
        n = len(self._levels)
        if n < 2: return
        xs  = [rect.origin.x + i * w / (n - 1) for i in range(n)]
        amp = h / 2 * 0.84
        top = [(xs[i], cy - self._levels[i] * amp) for i in range(n)]
        bot = [(xs[i], cy + self._levels[i] * amp) for i in range(n-1, -1, -1)]
        (BLUE if self._state == "recording" else DIM).setFill()
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
# Overlay panel — owns the NSPanel and drives the canvas
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
        panel.setContentView_(canvas)
        self._panel  = panel
        self._canvas = canvas

    def _updateLayout(self):
        if not self._panel: return
        text = self._transcript.strip()
        if not text:
            text_h = 0
        else:
            chars = int((PANEL_W - TX_PAD_H * 2) / 7.5)
            words = text.split(); lines = 1; cur = 0
            for w in words:
                if cur + len(w) + 1 > chars: lines += 1; cur = len(w)
                else: cur += len(w) + 1
            text_h = min(lines, MAX_LINES) * LINE_H + TX_PAD_V * 2
        total_h = PILL_H + text_h
        sf = NSScreen.mainScreen().frame()
        x = (sf.size.width - PANEL_W) / 2 + sf.origin.x
        y = sf.origin.y + BOTTOM
        self._panel.setFrame_display_(NSMakeRect(x, y, PANEL_W, total_h), True)
        self._canvas.setFrame_(NSMakeRect(0, 0, PANEL_W, total_h))
        if self._canvas:
            self._canvas.setText_(text)

    # -- animation timer ------------------------------------------------
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
            if self._canvas: self._canvas.setTimer_("")
            self._transcript = ""
            self._updateLayout()

    def setLevelsObj_(self, lvl):
        if self._canvas: self._canvas.setLevels_(lvl)

    def setTextObj_(self, text):
        self._transcript = text or ""
        self._updateLayout()

    # -- thread-safe wrappers -------------------------------------------
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
