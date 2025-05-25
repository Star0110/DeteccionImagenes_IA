import yaml

data = {
    'path': '../datasets/dataset_vehiculos',
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    'names': {
        0: 'car'
    }
}

with open('Programa5/datasets/dataset_vehiculos/data.yaml', 'w') as file:
    yaml.dump(data, file,
              default_flow_style=False,
              sort_keys=False)