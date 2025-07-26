import random


def generate_fake_bank_account(length=7):
    """Generate a fake bank account number with the specified length."""
    return ''.join(str(random.randint(0, 9)) for _ in range(length))


# Generate a fake bank account number
fake_account_number = generate_fake_bank_account()
# print("Fake Bank Account Number:", fake_account_number)
