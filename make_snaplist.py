for i in range(0,99):
    number = 0000
    number = number + i
    if len(str(number)) == 1:
        number = '000' + str(number)

    if len(str(number)) == 2:
        number = '00' + str(number)
    
    if len(str(number)) == 3:
        number = '0' + str(number)
    print(number)