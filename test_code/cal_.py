import random

number = []
people = 3
while len(number) < 3:
    ran = random.randint(1, 35)
    if ran != 23:
        number.append(ran)

print(number)
