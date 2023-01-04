import shutil

def open_cfile(fname):
    code_list = []
    with open (fname, "r") as reader:
        for line in reader.readlines():
            code_list.append(line)
    return code_list

def txt_to_c(src, dst):
    shutil.copyfile(src, dst)

def auto_edit(code_list):
    new_list = []
    for ct in code_list:
        list_ct = list(ct)
        list_ct.pop(0)
        list_ct.pop(0)
        for i, ls in enumerate(list_ct):
            if list_ct[i] == "t" and str.isalpha(list_ct[i+1]) == False:
                list_ct.insert(i+1, '[')
                a = i + 1
                for t, lst in enumerate(list_ct, start = a):
                    if list_ct[t] != '[' and str.isnumeric(list_ct[t]) == False:
                        list_ct.insert(t, ']')
                        break
  
        str_list = ''.join(list_ct)
        new_list.append(str_list)

    with open("C_Codes/temp.txt", "w") as f:

        f.write('#include <stdio.h>\n#include <math.h>\n\n') #create headers
        declaration = []
        for subscr in range(1, 7): #declare variables
            f.write(f'double q{subscr};\ndouble m{subscr};\ndouble a{subscr};\ndouble d{subscr};\ndouble Ixx{subscr};\ndouble Iyy{subscr};\ndouble Izz{subscr};\ndouble XC{subscr};\ndouble YC{subscr};\ndouble ZC{subscr};\ndouble g{subscr};\ndouble dq{subscr};\ndouble ddq{subscr};\n')
            declaration.append(f'q{subscr}, m{subscr}, a{subscr}, d{subscr}, Ixx{subscr}, Iyy{subscr}, Izz{subscr}, XC{subscr}, YC{subscr}, ZC{subscr}, g{subscr}, dq{subscr}, ddq{subscr}, ')
            #f.write(f'double main(q{subscr}, m{subscr}, a{subscr}, d{subscr}, Ixx{subscr}, Iyy{subscr}, Izz{subscr}, XC{subscr}, YC{subscr}, ZC{subscr}, double g{subscr}, double dq{subscr}, double ddq{subscr})')
        f.write(f'double t[2400];\ndouble A0[5][5];\ndouble XCtool;\ndouble YCtool;\ndouble ZCtool;\ndouble mtool;\ndouble Ixxtool;\ndouble Iyytool;\ndouble Izztool;\ndouble T_n_TCP_1_4;\ndouble T_n_TCP_2_4;\ndouble T_n_TCP_3_4;\n\n')

        str_declaration = ''.join(declaration)
        str_declaration = str_declaration[:-2]
        f.write(f'void main({str_declaration}){"{"}\n\n')

        for i in new_list: #write the edited codes into the file
            f.write(f'    {i}')
        f.write(f'\n{"}"}')
    
    txt_to_c("C_Codes/temp.txt", "C_Codes/EETransMatrix_edited.c")


cl = open_cfile("C_Codes/EETransMatrix.txt")
auto_edit(cl)