import random


def generate_fake_id(length=9):
    """Generate a fake ID number of specified length (default is 9)."""
    return ''.join(str(random.randint(0, 9)) for _ in range(length))


# Generate a fake ID
fake_id = generate_fake_id()
# print("Fake ID:", fake_id)
