"""Inline review/confirm editor — an editable field shown before paste.

When 'Review before paste' is on, the final transcript appears here. The user
can edit it, press Enter to confirm (→ paste + save as a verified sample), or
Esc to discard. Unlike the overlay pill, this panel CAN become key so it takes
keyboard input.
"""
import objc
from AppKit import (
    NSPanel, NSTextView, NSScrollView, NSScreen, NSColor, NSFont, NSView,
    NSMakeRect, NSBorderlessWindowMask, NSFocusRingTypeNone, NSApplication,
    NSBezierPath, NSString, NSFontAttributeName, NSForegroundColorAttributeName,
)
from Foundation import NSObject, NSMakePoint

W       = 560
MINH    = 60
MAXH    = 240
BOTTOM  = 64
PAD     = 16
CORNER  = 18.0
_BACKING = 2

def _c(r, g, b, a=1.0):
    return NSColor.colorWithRed_green_blue_alpha_(r, g, b, a)

BG    = _c(0.10, 0.10, 0.12, 0.99)
TEXT  = _c(0.95, 0.95, 0.97, 1.0)
HINT  = _c(0.55, 0.55, 0.60, 1.0)


class _KeyPanel(NSPanel):
    def canBecomeKeyWindow(self):
        return True


class _RoundedView(NSView):
    def isOpaque(self):
        return False

    def drawRect_(self, rect):
        hint = "Enter to send · Esc to cancel · Shift+Enter for newline"
        attrs = {NSFontAttributeName: NSFont.systemFontOfSize_(10.0),
                 NSForegroundColorAttributeName: HINT}
        NSString.stringWithString_(hint).drawAtPoint_withAttributes_(
            NSMakePoint(PAD, 5), attrs)


class _EditorTextView(NSTextView):
    def keyDown_(self, event):
        kc = event.keyCode()
        if kc == 36:  # Return
            if event.modifierFlags() & 0x00020000:   # Shift+Return → newline
                objc.super(_EditorTextView, self).keyDown_(event)
            elif self._owner:
                self._owner.submit()
            return
        if kc == 53:  # Esc
            if self._owner:
                self._owner.cancel()
            return
        objc.super(_EditorTextView, self).keyDown_(event)


class ReviewEditor(NSObject):
    def init(self):
        self = objc.super(ReviewEditor, self).init()
        if self is None:
            return None
        self._panel = None
        self._tv = None
        self._on_submit = None
        self._on_cancel = None
        return self

    def _setup(self):
        sf = NSScreen.mainScreen().frame()
        x = (sf.size.width - W) / 2 + sf.origin.x
        y = sf.origin.y + BOTTOM
        panel = _KeyPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, W, MINH), NSBorderlessWindowMask, _BACKING, False)
        panel.setLevel_(4)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(True)

        root = _RoundedView.alloc().initWithFrame_(NSMakeRect(0, 0, W, MINH))
        root.setWantsLayer_(True)
        root.layer().setCornerRadius_(CORNER)
        root.layer().setMasksToBounds_(True)
        root.layer().setBackgroundColor_(BG.CGColor())
        panel.setContentView_(root)

        scroll = NSScrollView.alloc().initWithFrame_(
            NSMakeRect(PAD, PAD + 16, W - PAD * 2, MINH - PAD * 2 - 16))
        scroll.setDrawsBackground_(False)
        scroll.setHasVerticalScroller_(True)
        scroll.setAutohidesScrollers_(True)

        tv = _EditorTextView.alloc().initWithFrame_(
            NSMakeRect(0, 0, W - PAD * 2, MINH))
        tv._owner = self
        tv.setDrawsBackground_(False)
        tv.setTextColor_(TEXT)
        tv.setFont_(NSFont.systemFontOfSize_(15.0))
        tv.setFocusRingType_(NSFocusRingTypeNone)
        tv.setInsertionPointColor_(_c(0.42, 0.78, 1.0, 1.0))
        tv.setRichText_(False)
        scroll.setDocumentView_(tv)
        root.addSubview_(scroll)
        self._scroll = scroll
        self._root = root

        # Hint label drawn by the rounded view
        self._tv = tv
        self._panel = panel

    def presentText_(self, text):
        # Callbacks are set as plain Python attributes by the caller
        # (passing Python callables through an ObjC selector mangles them).
        if self._panel is None:
            self._setup()
        self._tv.setString_(text or "")

        # Size to content
        sf = NSScreen.mainScreen().frame()
        lines = max(1, (text or "").count("\n") + 1 + len(text) // 70)
        h = min(MAXH, max(MINH, lines * 22 + PAD * 2 + 16))
        x = (sf.size.width - W) / 2 + sf.origin.x
        y = sf.origin.y + BOTTOM
        self._panel.setFrame_display_(NSMakeRect(x, y, W, h), True)
        self._root.setFrame_(NSMakeRect(0, 0, W, h))
        self._root.layer().setFrame_(self._root.bounds())
        self._scroll.setFrame_(NSMakeRect(PAD, PAD + 16, W - PAD * 2, h - PAD * 2 - 16))

        # Accessory (menu-bar) apps must activate to take keyboard focus
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        self._panel.makeKeyAndOrderFront_(None)
        self._panel.makeFirstResponder_(self._tv)
        # Select all so the user can immediately retype if they want
        self._tv.setSelectedRange_((0, len(text or "")))

    def submit(self):
        txt = self._tv.string()
        self._panel.orderOut_(None)
        if self._on_submit:
            self._on_submit(txt)   # caller re-activates target app + pastes

    def cancel(self):
        self._panel.orderOut_(None)
        NSApplication.sharedApplication().deactivate()
        if self._on_cancel:
            self._on_cancel()
