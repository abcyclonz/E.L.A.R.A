#!/usr/bin/env python3
"""
ELARA LCD Display - JHD160A 16x2 via I2C (PCF8574 backpack) on Raspberry Pi

  Row 0: [logo left][logo center][logo right]  ELARA   (status-aware logo)
  Row 1: [sp][base][sp] Pac-Man chases Ghost → eats it → ghost eyes return

Animation phases (cols 3-15, zone width = 13):
  Phase 0 CHASE     : Ghost flees right, Pac-Man chases from left  (12 steps)
  Phase 1 SCARED    : Ghost turns blue, Pac-Man closing in          (4 steps)
  Phase 2 EAT       : Pac-Man chomps: normal→half_eaten→feet        (3 steps)
  Phase 3 EYES      : Ghost eyes float back left, Pac-Man holds     (6 steps)
  Phase 4 RESET     : Pac-Man walks back left alone                 (6 steps)
  Then loops to Phase 0.

CGRAM slots:
  0 = left glyph      (arc or X)
  1 = center top
  2 = right glyph     (arc or X)
  3 = stick + base
  4 = pac_right_open  hot-swapped per mouth frame (phases 0-2)
  5 = pac_left_open   hot-swapped per mouth frame (phases 3-4)
  6 = ghost bitmap    hot-swapped per phase/step
  7 = (unused — reserved)

  Mouth cycle: open→half→closed→half (4 frames)

Wiring:
  I2C Module  ->  Raspberry Pi
  VCC         ->  5V  (Pin 2)
  GND         ->  GND (Pin 6)
  SDA         ->  GPIO 2 / SDA (Pin 3)
  SCL         ->  GPIO 3 / SCL (Pin 5)
"""

import smbus2
import time
import socket
import threading

# ── I2C config ──────────────────────────────────────────────────────────────
I2C_ADDR = 0x27
I2C_BUS  = 1

LCD_BACKLIGHT = 0x08
ENABLE        = 0x04
RS_BIT        = 0x01

LCD_CHR = RS_BIT
LCD_CMD = 0x00

LCD_LINE_1 = 0x80
LCD_LINE_2 = 0xC0
LCD_CLEAR  = 0x01
LCD_WIDTH  = 16

E_PULSE = 0.0005
E_DELAY = 0.0005

CHECK_INTERVAL = 5
FRAME_SPEED    = 0.20   # seconds per frame

# ── Logo bitmaps ─────────────────────────────────────────────────────────────
connected_left    = [0b01000,0b10010,0b10100,0b10101,0b10101,0b10100,0b10010,0b01000]
connected_right   = [0b00010,0b01001,0b00101,0b10101,0b10101,0b00101,0b01001,0b00010]
no_internet_left  = [0b00000,0b10001,0b01010,0b00100,0b01010,0b10001,0b00000,0b00000]
no_internet_right = [0b00000,0b10001,0b01010,0b00100,0b01010,0b10001,0b00000,0b00000]
center_top        = [0b00000,0b00000,0b00000,0b00000,0b01110,0b01110,0b00100,0b00100]
stick_base        = [0b00100,0b00100,0b00100,0b00100,0b00100,0b00100,0b01110,0b11111]

# ── Pac-Man bitmaps ──────────────────────────────────────────────────────────
# Facing RIGHT (→) — used while chasing ghost in phases 0-2
pac_right_open   = [0b01110,0b11110,0b11100,0b11000,0b11000,0b11100,0b11110,0b01110]
pac_right_half   = [0b01110,0b11111,0b11110,0b11100,0b11100,0b11110,0b11111,0b01110]
pac_right_closed = [0b01110,0b11111,0b11111,0b11111,0b11111,0b11111,0b11111,0b01110]

# Facing LEFT (←) — used while returning in phases 3-4
pac_left_open    = [0b01110,0b01111,0b00111,0b00011,0b00011,0b00111,0b01111,0b01110]
pac_left_half    = [0b01110,0b11111,0b01111,0b00111,0b00111,0b01111,0b11111,0b01110]
pac_left_closed  = [0b01110,0b11111,0b11111,0b11111,0b11111,0b11111,0b11111,0b01110]

# ── Ghost bitmaps ────────────────────────────────────────────────────────────
ghost_normal     = [0b01110,0b10101,0b11111,0b11111,0b11111,0b11111,0b10101,0b00000]
ghost_scared     = [0b01110,0b11011,0b11111,0b01110,0b11111,0b11111,0b01010,0b00000]
ghost_half_eaten = [0b00000,0b00000,0b00000,0b00000,0b11111,0b11111,0b10101,0b00000]
ghost_feet       = [0b00000,0b00000,0b00000,0b00000,0b00000,0b00000,0b10101,0b00000]
ghost_eyes       = [0b00000,0b01010,0b11110,0b00000,0b00000,0b00000,0b00000,0b00000]

# ── Animation layout ─────────────────────────────────────────────────────────
#
# Zone = cols 3-15 = 13 columns (indices 0-12 within zone).
#
# Phase 0 CHASE (12 steps, both move RIGHT):
#   Pac-Man : col 0 → 10   (leads)
#   Ghost   : col 2 → 12   (flees ahead of Pac-Man, 2 cols in front)
#   Pac-Man faces RIGHT, ghost is normal.
#
# Phase 1 SCARED (4 steps, both slow/hold):
#   Pac-Man : col 10, closing to 11
#   Ghost   : col 12 → 11  (backs into right wall)
#   Ghost turns SCARED.
#
# Phase 2 EAT (3 steps):
#   Pac-Man moves onto ghost position, ghost bitmap cycles:
#   step 0: ghost_normal (just before bite), pac at 11
#   step 1: ghost_half_eaten,               pac at 12
#   step 2: ghost_feet,                     pac stays 12
#
# Phase 3 EYES (6 steps):
#   Ghost eyes float from col 12 back to col 0 (2 cols per step).
#   Pac-Man holds at col 12, faces LEFT (turning around).
#
# Phase 4 RESET (6 steps):
#   Pac-Man walks LEFT from col 12 → 0.
#   No ghost drawn.
#
ZONE_START      = 3
ZONE_WIDTH      = 13
STEPS_CHASE     = 12
STEPS_SCARED    = 4
STEPS_EAT       = 3
STEPS_EYES      = 6
STEPS_RESET     = 6

bus = smbus2.SMBus(I2C_BUS)
lcd_lock = threading.Lock()


# ── Connectivity ─────────────────────────────────────────────────────────────
def is_connected():
    try:
        socket.setdefaulttimeout(2)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        return True
    except (socket.error, OSError):
        return False


# ── I2C / LCD primitives ─────────────────────────────────────────────────────
def i2c_write(data):
    bus.write_byte(I2C_ADDR, data | LCD_BACKLIGHT)

def lcd_toggle_enable(data):
    time.sleep(E_DELAY)
    i2c_write(data | ENABLE)
    time.sleep(E_PULSE)
    i2c_write(data & ~ENABLE)
    time.sleep(E_DELAY)

def lcd_byte(bits, mode):
    high = mode | (bits & 0xF0)
    low  = mode | ((bits << 4) & 0xF0)
    i2c_write(high); lcd_toggle_enable(high)
    i2c_write(low);  lcd_toggle_enable(low)

def lcd_init():
    time.sleep(0.02)
    for _ in range(3):
        i2c_write(0x30); lcd_toggle_enable(0x30); time.sleep(0.005)
    i2c_write(0x20); lcd_toggle_enable(0x20); time.sleep(0.001)
    lcd_byte(0x28, LCD_CMD)
    lcd_byte(0x0C, LCD_CMD)
    lcd_byte(0x06, LCD_CMD)
    lcd_byte(LCD_CLEAR, LCD_CMD)
    time.sleep(0.002)

def lcd_load_custom_char(slot, char_map):
    lcd_byte(0x40 | (slot << 3), LCD_CMD)
    for row in char_map:
        lcd_byte(row, LCD_CHR)

def lcd_set_cursor(row, col):
    addr = (LCD_LINE_1 if row == 0 else LCD_LINE_2) + col
    lcd_byte(addr, LCD_CMD)

def lcd_string(text):
    for ch in text:
        lcd_byte(ord(ch), LCD_CHR)

def lcd_custom_char(slot):
    lcd_byte(slot, LCD_CHR)


# ── Load all CGRAM slots ──────────────────────────────────────────────────────
def load_chars(connected):
    left  = connected_left  if connected else no_internet_left
    right = connected_right if connected else no_internet_right
    lcd_load_custom_char(0, left)
    lcd_load_custom_char(1, center_top)
    lcd_load_custom_char(2, right)
    lcd_load_custom_char(3, stick_base)
    lcd_load_custom_char(4, pac_right_open)  # hot-swapped each frame
    lcd_load_custom_char(5, pac_left_open)   # hot-swapped each frame
    lcd_load_custom_char(6, ghost_normal)    # hot-swapped per phase/step


# ── Draw row 0 ────────────────────────────────────────────────────────────────
def draw_row0():
    label   = "ELARA"
    padding = (13 - len(label)) // 2
    text    = (" " * padding + label).ljust(13)
    lcd_set_cursor(0, 0)
    lcd_custom_char(0)
    lcd_custom_char(1)
    lcd_custom_char(2)
    lcd_string(text)


# ── Draw static antenna base on row 1 cols 0-2 ───────────────────────────────
def draw_row1_static():
    lcd_set_cursor(1, 0)
    lcd_string(" ")
    lcd_custom_char(3)
    lcd_string(" ")


# ── Hot-swap a CGRAM slot then restore DDRAM pointer to row 1 ────────────────
def hot_swap(slot, bitmap):
    lcd_load_custom_char(slot, bitmap)
    lcd_byte(LCD_LINE_2, LCD_CMD)


# ── Draw one animation frame ──────────────────────────────────────────────────
def draw_anim_frame(pac_pos, ghost_pos, pac_slot, ghost_slot):
    """
    pac_pos / ghost_pos : position index 0-12 within zone.
                          Pass None to omit a character.
    pac_slot  : CGRAM slot 4 (right-facing) or 5 (left-facing)
    ghost_slot: CGRAM slot 6
    """
    buf = [None] * ZONE_WIDTH

    if ghost_pos is not None and 0 <= ghost_pos < ZONE_WIDTH:
        buf[ghost_pos] = ghost_slot

    # Pac-Man overwrites ghost if they share a cell (eat moment)
    if pac_pos is not None and 0 <= pac_pos < ZONE_WIDTH:
        buf[pac_pos] = pac_slot

    lcd_set_cursor(1, ZONE_START)
    for item in buf:
        if item is None:
            lcd_string(" ")
        else:
            lcd_custom_char(item)


# ── Mouth cycle tables ────────────────────────────────────────────────────────
RIGHT_MOUTH = [pac_right_open, pac_right_half, pac_right_closed, pac_right_half]
LEFT_MOUTH  = [pac_left_open,  pac_left_half,  pac_left_closed,  pac_left_half]


# ── Animation thread ──────────────────────────────────────────────────────────
def anim_thread(stop_event):
    phase = 0
    step  = 0
    frame = 0

    while not stop_event.is_set():
        mf = frame % 4

        with lcd_lock:

            # ── Phase 0: Chase — Pac-Man chases ghost rightward ───────────
            if phase == 0:
                t         = step / max(STEPS_CHASE - 1, 1)
                pac_pos   = round(t * 10)        # 0 → 10
                ghost_pos = round(2 + t * 10)    # 2 → 12

                hot_swap(4, RIGHT_MOUTH[mf])
                hot_swap(6, ghost_normal)
                draw_anim_frame(pac_pos, ghost_pos, 4, 6)

                step += 1
                if step >= STEPS_CHASE:
                    phase = 1; step = 0

            # ── Phase 1: Scared — ghost backs against wall ─────────────────
            elif phase == 1:
                pac_pos   = 10 + step            # 10 → 13 (clipped by draw)
                ghost_pos = 12                   # pinned at right wall

                hot_swap(4, RIGHT_MOUTH[mf])
                hot_swap(6, ghost_scared)
                draw_anim_frame(min(pac_pos, 11), ghost_pos, 4, 6)

                step += 1
                if step >= STEPS_SCARED:
                    phase = 2; step = 0

            # ── Phase 2: Eat — Pac-Man chomps ghost ───────────────────────
            elif phase == 2:
                eat_bitmaps = [ghost_normal, ghost_half_eaten, ghost_feet]
                pac_pos     = 11 + step          # 11, 12, 12

                hot_swap(4, RIGHT_MOUTH[mf])
                hot_swap(6, eat_bitmaps[step])
                # On final step pac and ghost share col 12 — pac overwrites
                draw_anim_frame(min(pac_pos, 12), 12, 4, 6)

                step += 1
                if step >= STEPS_EAT:
                    phase = 3; step = 0

            # ── Phase 3: Eyes — ghost eyes float left, pac turns around ───
            elif phase == 3:
                eye_pos = max(12 - step * 2, 0)  # 12 → 2 → 0

                hot_swap(5, LEFT_MOUTH[mf])
                hot_swap(6, ghost_eyes)
                draw_anim_frame(12, eye_pos, 5, 6)

                step += 1
                if step >= STEPS_EYES:
                    phase = 4; step = 0

            # ── Phase 4: Reset — Pac-Man walks back left, no ghost ────────
            elif phase == 4:
                t       = step / max(STEPS_RESET - 1, 1)
                pac_pos = round(12 - t * 12)     # 12 → 0

                hot_swap(5, LEFT_MOUTH[mf])
                draw_anim_frame(pac_pos, None, 5, 6)

                step += 1
                if step >= STEPS_RESET:
                    phase = 0; step = 0

        frame += 1
        time.sleep(FRAME_SPEED)


# ── Re-render row 0 on status change ─────────────────────────────────────────
def render_row0(connected):
    with lcd_lock:
        load_chars(connected)
        lcd_byte(LCD_LINE_1, LCD_CMD)
        draw_row0()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    try:
        lcd_init()

        last_status = is_connected()
        load_chars(last_status)

        lcd_byte(LCD_CLEAR, LCD_CMD)
        time.sleep(0.002)

        draw_row0()
        draw_row1_static()

        stop_event = threading.Event()
        anim = threading.Thread(target=anim_thread, args=(stop_event,), daemon=True)
        anim.start()

        print("ELARA LCD running — press Ctrl+C to exit.")
        while True:
            time.sleep(CHECK_INTERVAL)
            status = is_connected()
            if status != last_status:
                print(f"Status: {'CONNECTED' if status else 'NO INTERNET'}")
                render_row0(status)
                last_status = status

    except KeyboardInterrupt:
        stop_event.set()
        with lcd_lock:
            lcd_byte(LCD_CLEAR, LCD_CMD)
        print("\nDisplay cleared. Goodbye.")
    finally:
        bus.close()


if __name__ == "__main__":
    main()