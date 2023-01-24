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
        list_ct.remove(";")
        str_list = ''.join(list_ct)
        new_list.append(str_list)

    with open("C:\Codes\master_thesis\C_Codes/temp.txt", "w") as f:
        for i in new_list: #write the edited codes into the file
            f.write(f'    {i}')

    txt_to_c("C:\Codes\master_thesis\C_Codes/temp.txt", "C:\Codes\master_thesis\C_Codes/tau.py")



cl = open_cfile("C:\Codes\master_thesis\C_Codes\Tau_Joint.txt")
auto_edit(cl)