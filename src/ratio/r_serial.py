import serial
import time
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

SERIAL_PORT = os.getenv("SERIAL_PORT", "COM4")
BAUD_RATE = int(os.getenv("BAUD_RATE", 19200))
DURATION = int(os.getenv("DURATION", 5))


# Open the serial port
def open_serial_connection():
    try:
        ser = serial.Serial(
            SERIAL_PORT, BAUD_RATE, timeout=1
        )  # Use serial.Serial, not serial
        print(f"Opened serial port {SERIAL_PORT} with baud rate {BAUD_RATE}")
        return ser
    except Exception as e:
        print(f"Error opening serial port: {str(e)}")
        return None


# Function to measure data from the Serial Port
def measure_serial_data(ser):
    start_time = time.time()
    sample_count = 0

    while time.time() - start_time < DURATION:
        if ser.in_waiting > 0:  # If there is data to read
            value = ser.readline().decode("utf-8").strip()  # Read a line and decode it
            if value:
                sample_count += 1
    return sample_count  # Return the number of samples received


def close_serial_connection(ser):
    if ser and ser.is_open:
        ser.close()
        print(f"Closed serial port {SERIAL_PORT}")


# Main logic
if __name__ == "__main__":
    # Open the serial connection
    ser = open_serial_connection()

    if ser:
        try:
            # Measure data for the specified duration
            serial_samples = measure_serial_data(ser)

            print(f"\nMessage Received: {serial_samples}")

            print(f"\nDuration: {DURATION}")
            print(f"Average : {(DURATION/serial_samples)*1000}ms\n")
        except KeyboardInterrupt:
            print("Measurement interrupted.")
        finally:
            # Close the serial connection
            close_serial_connection(ser)
