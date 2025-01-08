import random
import string


def generate_fake_driver_license():
    """Generate a fake driver's license number with a generic format."""
    state_code = ''.join(random.choices(string.ascii_uppercase, k=2))  # Two uppercase letters
    year = str(random.randint(1950, 2025))[-2:]  # Last two digits of a year
    serial = ''.join(random.choices(string.digits, k=6))  # Six digits

    return f"{state_code}-{year}-{serial}"


# Generate a fake driver's license number
fake_license = generate_fake_driver_license()
# print("Fake Driver's License:", fake_license)
