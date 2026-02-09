#Requires AutoHotkey v2.0

; Right Shift + Up Arrow → Move browser tab to the right
; Right Shift + Down Arrow → Move browser tab to the left
; Works in Chrome, Edge, Brave, and other Chromium browsers

RShift & Up:: {
    Send "^+{PgDn}"
}

RShift & Down:: {
    Send "^+{PgUp}"
}
