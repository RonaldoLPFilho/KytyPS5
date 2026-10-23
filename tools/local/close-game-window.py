#!/usr/bin/env python3
"""Request normal window close for an explicitly identified emulator PID.

This lets the emulator save its pipeline cache. It does not send a kill signal.
"""
import argparse
import ctypes as C
import os
from pathlib import Path
from debug_cpu_policy import start_ticks
from game_recording import find_game_window

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--pid', type=int, required=True)
a = p.parse_args()
proc = Path('/proc') / str(a.pid)
if Path(os.readlink(proc / 'exe')).name.removesuffix(' (deleted)') != 'kyty_emulator':
    p.error('Expected a live emulator')
identity = start_ticks(a.pid)
window = find_game_window(a.pid)
if not window:
    p.error('Owned window not found')
class Data(C.Union):
    _fields_ = [('b', C.c_char * 20), ('s', C.c_short * 10), ('l', C.c_long * 5)]
class ClientMessage(C.Structure):
    _fields_ = [('type', C.c_int), ('serial', C.c_ulong), ('send_event', C.c_int),
                ('display', C.c_void_p), ('window', C.c_ulong), ('message_type', C.c_ulong),
                ('format', C.c_int), ('data', Data)]
class Event(C.Union):
    _fields_ = [('client', ClientMessage), ('padding', C.c_long * 24)]
x = C.CDLL('libX11.so.6')
x.XOpenDisplay.restype = C.c_void_p
x.XInternAtom.argtypes = [C.c_void_p, C.c_char_p, C.c_int]; x.XInternAtom.restype = C.c_ulong
x.XSendEvent.argtypes = [C.c_void_p, C.c_ulong, C.c_int, C.c_long, C.POINTER(Event)]
x.XFlush.argtypes = [C.c_void_p]; x.XCloseDisplay.argtypes = [C.c_void_p]
d = x.XOpenDisplay(None)
if not d:
    raise RuntimeError('No X11 display')
try:
    event = Event()
    event.client.type, event.client.display, event.client.window = 33, d, window
    event.client.message_type = x.XInternAtom(d, b'WM_PROTOCOLS', 0)
    event.client.format = 32
    event.client.data.l[0] = x.XInternAtom(d, b'WM_DELETE_WINDOW', 0)
    if start_ticks(a.pid) != identity:
        raise RuntimeError('Process identity changed')
    if not x.XSendEvent(d, window, 0, 0, C.byref(event)):
        raise RuntimeError('Window close request failed')
    x.XFlush(d)
    print(f'Requested normal window close: PID {a.pid}, start {identity}, window {window:#x}')
finally:
    x.XCloseDisplay(d)
