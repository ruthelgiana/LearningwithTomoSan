'''print("Hello World!")'''

'''s1, s2 = "Python " , "is great."
print(s1 + s2)'''

'''s= "Python"
print(s*4)'''

'''s="Jack"
print(f"My name is {s}.")'''

'''s1="Jack"
s2=10
print(f"Name={s1}; Age={s2}")'''

'''s1="Python"
s2="great "
print(f"{s1} is {s2*4}")'''

'''a = b = c = 1
print(a)
print(b)
print(c)'''

'''a, b, c = 1, 2.5, 'abc def'
print(a)
print(b)
print(c)'''

'''name = input("What is your name?")
print(f"Your name is {name}.")'''

'''name = input("What is your name?")
age = int(input("How old are you?"))
print(f"""Your name is {name}.
You are {age} years old.""")'''

'''year = int(input("Year of birth?"))
age = 2021 - year
print(f"You are {age} years old.")'''

name = int(input("How many pieces of chicken do you want to buy?"))
Total = 12500*name
print(f"Total Rp{Total}.0 ")
discount = int(0.17*Total)
print((f"You get 17^% discount, so you should pay Rp{discount}.0"))
Afterdiscount = Total - discount
print(f"The total price will be Rp{discount}.0")
VAT = int(0.1 * Afterdiscount )
print((f"VAT 10% will be Rp{VAT}.0"))
TotalPayment = int(Afterdiscount + VAT)
print((f"Total payment is Rp{TotalPayment}.0"))