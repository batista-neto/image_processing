import os
import random

# Diretórios principais
image_dir = 'ade/JPEGImages'
mask_dir = 'ade/SegmentationClass'
splits_base = 'dataset/splits/ade'
val_txt = os.path.join(splits_base, 'val.txt')

# Configurações dos splits
split_configs = {
    '1_8': 1/8,
    '1_4': 1/4,
    '183': 183,
    '366': 366
}

# Garante a existência da pasta base
os.makedirs(splits_base, exist_ok=True)

# Lista imagens válidas com máscara correspondente
valid_files = []
for img_name in sorted(os.listdir(image_dir)):
    if not img_name.endswith('.jpg'):
        continue
    base = os.path.splitext(img_name)[0]
    mask_path = os.path.join(mask_dir, base + '.png')
    if os.path.exists(mask_path):
        valid_files.append(base)

# Gera val.txt
with open(val_txt, 'w') as f:
    for base in valid_files:
        f.write(f"ade/JPEGImages/{base}.jpg ade/SegmentationClass/{base}.png\n")
print(f"[✓] val.txt criado com {len(valid_files)} pares válidos")

# Embaralha os arquivos para reprodutibilidade
random.seed(42)
random.shuffle(valid_files)

# Gera os splits
for split_name, amount in split_configs.items():
    split_0_dir = os.path.join(splits_base, split_name, 'split_0')
    os.makedirs(split_0_dir, exist_ok=True)

    if isinstance(amount, float):
        split_size = int(len(valid_files) * amount)
    else:
        split_size = int(amount)

    labeled = valid_files[:split_size]
    unlabeled = valid_files[split_size:]

    def to_line(base):
        return f"ade/JPEGImages/{base}.jpg ade/SegmentationClass/{base}.png\n"

    with open(os.path.join(split_0_dir, 'labeled.txt'), 'w') as f:
        f.writelines([to_line(b) for b in labeled])

    with open(os.path.join(split_0_dir, 'unlabeled.txt'), 'w') as f:
        f.writelines([to_line(b) for b in unlabeled])

    print(f"[✓] Split {split_name}/split_0 criado: {len(labeled)} labeled, {len(unlabeled)} unlabeled")
