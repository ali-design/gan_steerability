BATCH_SIZE = 4

DIM_Z = 512 # used only for setting default arg value

MODEL_ROOT = 'resources/pgan_pretrained/'

net_info = dict(
    celebahq=dict(
        path=MODEL_ROOT + 'karras2018iclr-celebahq-1024x1024.pkl',
        img_size=1024,
        coco_id=None,
        is_face=True
    ),
)



