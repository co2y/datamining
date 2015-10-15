import itertools

MIN_SUP = 0.144
TRANSACTIONS_NUMBER = 1000
ITEMS_NUMBER = 11
MOD_NUMBER = 3
NODE_MAX_ITEMS = 5
F = list()
transactions_list = list()


class Node:
    def __init__(self, is_leaf_node, layer_number):
        self.is_leaf_node = is_leaf_node
        self.layer_number = layer_number
        self.candidates = list()
        self.child = list()

    def add_child(self):
        self.is_leaf_node = False
        for i in range(MOD_NUMBER):
            self.child.append(Node(True, self.layer_number + 1))


def get_all_transactions():
    with open("assignment2-data.txt", 'r') as f:
        f.readline()
        line = f.readline()
        while line:
            transactions_list.append([chr(x + 97) for x, y in enumerate(line.strip().split(" ")) if y == '1'])
            line = f.readline()
    return transactions_list


def get_f1():
    F.append(dict())
    for each_transaction in transactions_list:
        for each_item in each_transaction:
            if each_item not in F[0]:
                F[0][each_item] = 1
            else:
                F[0][each_item] += 1
    F[0] = {key: value for key, value in F[0].items() if value / TRANSACTIONS_NUMBER >= MIN_SUP}


def get_fk():
    for i in range(ITEMS_NUMBER):
        if not F[i]:
            break
        generate_ck(i)
        prune_ck(i)
        support_counting(i)


def generate_ck(k):
    F.append(dict())
    sorted_keys = sorted(F[k].keys())
    for i, item_i in enumerate(sorted_keys[:-1]):
        for item_j in sorted_keys[i + 1:]:
            if item_i[:-1] == item_j[:-1]:
                temp_item = item_i + item_j[-1]
                F[k + 1][temp_item] = 0
            else:
                break


def prune_ck(k):
    for each_item in F[k + 1]:
        combination_list = list(itertools.combinations(each_item, k + 1))
        for each_combination in combination_list:
            subset = "".join(each_combination)
            if subset not in F[k]:
                del F[k + 1][subset]
                break


def support_counting(k):
    sorted_keys = sorted(F[k + 1].keys())
    root = Node(False, 0)
    root.add_child()
    for each_items in sorted_keys:
        mod_id = ord(each_items[0]) % MOD_NUMBER
        now_node = root.child[mod_id]
        while 1:
            if now_node.is_leaf_node:
                now_node.candidates.append(each_items)
                if now_node.candidates.__len__() > NODE_MAX_ITEMS and now_node.layer_number < k + 2:
                    now_node.add_child()
                    for each_candidate in now_node.candidates:
                        now_node.child[ord(each_candidate[now_node.layer_number]) % MOD_NUMBER].candidates.append(
                            each_candidate)
                break
            else:
                now_node = now_node.child[ord(each_items[now_node.layer_number]) % MOD_NUMBER]
    for each_transaction in transactions_list:
        node_list = list()
        node_list.append(root)
        while node_list:
            now_node = node_list.pop(0)
            if now_node.is_leaf_node:
                for each_candidate in now_node.candidates:
                    if set(each_candidate).issubset(set(each_transaction)):
                        F[k + 1][each_candidate] += 1
            else:
                for each_node in now_node.child:
                    node_list.append(each_node)
    F[k + 1] = {key: value for key, value in F[k + 1].items() if value / TRANSACTIONS_NUMBER >= MIN_SUP}


def print_result():
    for i in range(ITEMS_NUMBER):
        if not F[i]:
            break
        for each_key in sorted(F[i].keys()):
            temp_str = ""
            for each_chr in each_key:
                temp_str += str(ord(each_chr) - 96) + " "
            print('%s%.3f' % (temp_str, round(F[i][each_key] / TRANSACTIONS_NUMBER, 3)))


if __name__ == '__main__':
    get_all_transactions()
    get_f1()
    get_fk()
    print_result()