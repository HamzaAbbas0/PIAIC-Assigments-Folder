
                                        # Variables and String
'''a="Hello Welcome to the Artifical Intelligence course"
print(a)
quote=("“Everything is theoretically impossible, until it  done.”")
print(quote)'''

                                       # Calculate Area Of Circle
'''radius= float(input("Enter the Radius of Cirlce : "))
result=3.14*(radius**2)
print(result)'''

                                       # Check Number either positive, negative or zero::
'''number=int(input("Enter the number to find postive negative and zero : "))
if number >0:
    print("your Given Number is",number,"And its a Postive Number")
elif number ==0:
    print("you Enter the zero number")
else:
    print("your Given Number is", number, "And its a Negative Number")'''

                                       # Vowel Tester
'''fndvowel=input("Enter the Character to find the Vowels Numbers :")
list=["A","a","E","e","I","i","O","o","U","u"]
if fndvowel in list:
    print(fndvowel,": Its a vowvel Number")
else:
    print(fndvowel,": Its not a Vowvel Number")'''

                                     #  BMI-Calculator
'''height =int(input("Enter The Height in cm :"))
weight =int(input("Enter Weight in Kg :"))

x=height /100
x1 =x**2
result =weight /x1
print(result)'''
                                     # LIST
'''names=["Hamza","Ali","Ahmed","Abdullah","Hassan","faizan"]
print("Hello",names[0],"Where i you see you after a long time")
print("Hello",names[1],"Where i you see you after a long time")
print("Hello",names[2],"Where i you see you after a long time")
print("Hello",names[3],"Where i you see you after a long time")
print("Hello",names[4],"Where i you see you after a long time")
print("Hello",names[5],"Where i you see you after a long time")'''

                                  # Favourite Dishes
'''dishes=["baryani","korma","palao","fruit-chat","bindi","Nihari","qeema","kofty","Chicken Tikka"]
#x=slice(3)
#print(dishes[x])
print(dishes[0:3])
print(dishes[3:6])
print(dishes[6:9])'''

                                # Friend Dishes part 2
'''dishes=["baryani","korma","palao","fruit-chat","bindi","Nihari","qeema","kofty","Chicken Tikka"]
dishes.append("Haleem")
frnddishes=["baryani","korma","palao","fruit-chat","bindi","Nihari","qeema","kofty","Chicken Tikka"]
frnddishes.append("Samosas")
for i in dishes:
    x=i
    print("My favourite dishes are",x)
for i in frnddishes:
    y=i
    print("My friend-favourite dishes are",y)'''