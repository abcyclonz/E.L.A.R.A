import secrets
import requests
import time
import subprocess
from RPLCD.i2c import CharLCD
from gpiozero import Button
from signal import pause

# --- CONFIG ---
BACKEND_URL = "http://10.235.223.193:8001"
SHARED_SECRET = "your_shared_secret_key_123"
USER_ID = "USER_01"
OTP_LIFETIME = 120
REFRESH_BUFFER = 30
POLL_INTERVAL = 10

# --- LCD INIT ---
try:
    lcd = CharLCD('PCF8574', 0x27)
except Exception as e:
    print(f"Hardware Error: Could not find LCD at 0x27. {e}")
    lcd = None

button = Button(17, hold_time=1)


def generate_otp() -> str:
    """Generate a 6-digit numeric OTP."""
    return "".join(secrets.choice("0123456789") for _ in range(6))


def update_lcd(line1: str, line2: str = "") -> None:
    if lcd:
        lcd.clear()
        lcd.write_string(line1[:16])
        if line2:
            lcd.cursor_pos = (1, 0)
            lcd.write_string(line2[:16])
    print(f"LCD L1: {line1} | L2: {line2}")


def update_lcd_line2(line2: str) -> None:
    """Update only line 2 without clearing line 1."""
    if lcd:
        lcd.cursor_pos = (1, 0)
        lcd.write_string(f"{line2[:16]:<16}")
    print(f"LCD L2: {line2}")


def sync_otp(otp: str) -> bool:
    """Push OTP to backend. Returns True on success."""
    headers = {"X-API-KEY": SHARED_SECRET}
    payload = {
        "user_id": USER_ID,
        "otp": otp,
        "expires_in": OTP_LIFETIME,
    }
    try:
        response = requests.post(
            f"{BACKEND_URL}/sync-otp",
            json=payload,
            headers=headers,
            timeout=10
        )
        if response.status_code == 200:
            return True
        update_lcd("Sync Failed", f"HTTP {response.status_code}")
        print(f"Sync failed — {response.status_code}: {response.text}")
        return False

    except requests.exceptions.ConnectionError:
        update_lcd("Conn Error", "Check Network")
        print("Connection error: backend unreachable.")
    except requests.exceptions.Timeout:
        update_lcd("Timeout", "Retrying soon")
        print("Request timed out.")
    except Exception as e:
        update_lcd("Unknown Error", "See terminal")
        print(f"Unexpected error: {e}")

    return False


def check_connection_status() -> bool:
    """Ask backend once if frontend has verified. Returns True if verified."""
    headers = {"X-API-KEY": SHARED_SECRET}
    try:
        response = requests.get(
            f"{BACKEND_URL}/connection-status",
            headers=headers,
            params={"user_id": USER_ID},
            timeout=5
        )
        if response.status_code == 200:
            return response.json().get("verified", False)
    except Exception as e:
        print(f"Status check error: {e}")
    return False


def wait_for_verification(otp: str) -> bool:
    """
    Poll backend until OTP is verified or expires.
    GPIO 17 button is ignored during this entire period via a no-op handler.
    - If verified: handler stays as no-op (button remains disabled).
    - If expired:  handler reset to None so wait_for_button_press can rearm it.
    """
    button.when_held = lambda: None  # no-op — swallow any press during verification

    deadline = time.time() + OTP_LIFETIME - REFRESH_BUFFER
    update_lcd(f"OTP: {otp}", "Exp in 5m00s")

    while time.time() < deadline:
        if check_connection_status():
            # Verified — leave when_held as no-op, button stays disabled
            return True

        for _ in range(POLL_INTERVAL):
            if time.time() >= deadline:
                break
            remaining = int(deadline - time.time())
            mins, secs = divmod(remaining, 60)
            update_lcd_line2(f"Exp in {mins}m{secs:02d}s")
            time.sleep(1)

    # Expired — re-enable button so wait_for_button_press can rearm it
    button.when_held = None
    return False


def wait_for_button_press() -> None:
    """
    Block until the button is held for 1 second.
    Shows a prompt on the LCD while waiting.
    """
    update_lcd("Hold button to", "generate OTP")
    print("Waiting for button hold...")

    pressed = False

    def on_held():
        nonlocal pressed
        pressed = True

    button.when_held = on_held

    while not pressed:
        time.sleep(0.1)

    button.when_held = None  # clean up handler


def run_otp_loop() -> None:
    """
    Main loop:
    1. Wait for button hold
    2. Generate OTP
    3. Sync OTP to backend
    4. Wait for frontend verification
    5. On verified → show success and exit (one-shot)
    6. On expiry → go back to step 1 and wait for button again
    """
    while True:
        # Step 1: wait for button press
        wait_for_button_press()

        # Step 2: generate
        otp = generate_otp()
        print(f"Generated OTP: {otp}")

        # Step 3: sync
        update_lcd("Syncing...", "Please wait")
        if not sync_otp(otp):
            print("Sync failed. Retrying in 5s...")
            update_lcd("Sync Failed", "Try again soon")
            time.sleep(5)
            continue

        print(f"OTP synced: {otp} — waiting for frontend verification...")

        # Step 4: wait for verification
        verified = wait_for_verification(otp)

        if verified:
            update_lcd("Verified!", "Access Granted")
            time.sleep(2)
            update_lcd("ELARA:", "HI")
            print("Frontend verified the OTP. Connection established!")
            try:
                # Option A: Run and wait for it to finish
                subprocess.run(["python", "rasp-disp.py"]) 
                
                
            except Exception as e:
                print(f"Failed to launch script: {e}")
            break

        else:
            print("OTP expired without verification.")
            update_lcd("OTP Expired", "Hold btn again")
            time.sleep(2)


if __name__ == "__main__":
    run_otp_loop()