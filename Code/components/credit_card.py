import random


def luhn_checksum(card_number):
    """Calculate the Luhn checksum for a given card number."""
    def digits_of(n):
        return [int(d) for d in str(n)]
    digits = digits_of(card_number)
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    checksum = sum(odd_digits)
    for d in even_digits:
        checksum += sum(digits_of(d * 2))
    return checksum % 10


def is_luhn_valid(card_number):
    """Check if a card number is valid according to the Luhn algorithm."""
    return luhn_checksum(card_number) == 0


def generate_card_number(prefix, length):
    """Generate a valid card number with a given prefix and length."""
    number = [int(x) for x in str(prefix)]
    while len(number) < (length - 1):
        number.append(random.randint(0, 9))
    checksum = luhn_checksum(int("".join(map(str, number))) * 10)
    check_digit = (10 - checksum) % 10
    number.append(check_digit)
    return ''.join(map(str, number))


# Example usage:
prefix = 4580  # Visa prefix
length = 16  # Typical length for Visa cards
fake_card = generate_card_number(prefix, length)
# print("Generated fake card number:", fake_card)
