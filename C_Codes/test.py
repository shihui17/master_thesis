code_list = []
new_list = []

with open ("C_Codes\Tau_Joint.txt", "r") as reader:
    for line in reader.readlines():
        code_list.append(line)

new_list = []
for ct in code_list:
    list_ct = list(ct)
    list_ct.pop(0)
    list_ct.pop(0)
    print(list_ct)
    break
