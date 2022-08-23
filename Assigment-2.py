mth=int(input("please enter math number"))
eng=int(input("please enter eng number"))
urd=int(input("please enter urd number"))
isl=int(input("please enter isl number"))
phy=int(input("please enter phy number"))
chem=int(input("please enter chem number"))
if mth < 33 or eng <33 or urd <33 or isl <33 or phy <33 or chem <33:
    print("you fail in subject")

else:
    add=mth+eng+urd+isl+phy+chem
    print("Obtained Number",add)
    prctg= (add*100)/600
    print(prctg,"%")
    if prctg >=80:
        print("you got A+ Grade")
    elif prctg >=70 and prctg<=79:
        print("you got A Grade")
    elif prctg >=60 and prctg<=69:
        print("you got B Grade")
    elif prctg >=50 and prctg<=59:
        print("you got C Grade")
    elif prctg >=40 and prctg<=49:
        print("you got D Grade")
    else:
        print("sorry you are failed")