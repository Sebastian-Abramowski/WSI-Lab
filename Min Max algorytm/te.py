list_of_lists = [[1, 2, 3, 3, 3, 3], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]]

last_four_elements = [set(sublist[-4:]) for sublist in list_of_lists]

print(last_four_elements)
