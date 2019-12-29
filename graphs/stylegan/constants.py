BATCH_SIZE = 4

DIM_Z = 512 # used only for setting default arg value

net_info = dict(
    cars=dict(
        url='https://drive.google.com/uc?id=1MJ6iCfNtMIRicihwRorsM3b7mmtmK9c3',
        img_size=512,
        coco_id=3,
        pascal_id=7
    ),
    cats=dict(
        url='https://drive.google.com/uc?id=1MQywl0FNt6lHu8E_EUqnRbviagS7fbiJ',
        img_size=256,
        coco_id=17,
        pascal_id=8
    ),
    bedrooms=dict(
        url='https://drive.google.com/uc?id=1MOSKeGF0FJcivpBI7s63V9YHloUTORiF',
        img_size=256,
        coco_id=None,
        pascal_id=None
    ),
    celebahq=dict(
        url='https://drive.google.com/uc?id=1MGqJl28pN4t7SAtSrPdSRJSQJqahkzUf',
        img_size=1024,
        coco_id=None,
        pascal_id=None,
        is_face=True
    ),
    ffhq=dict(
        url='https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ',
        img_size=1024,
        coco_id=None,
        pascal_id=None,
        is_face=True
    ),
)



