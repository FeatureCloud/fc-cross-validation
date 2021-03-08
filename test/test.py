from app.algo import read_data, create_splits

dataset = read_data('client.csv', sep=',')
print(dataset)

create_splits(dataset, None, 5, False, True, 42, '.')
