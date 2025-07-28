import os
import random

# Caminhos
image_dir = "pascal/JPEGImages"
mask_dir = "pascal/SegmentationClass"
split_base = "dataset/splits/pascal/1_16/split_0"
val_path = "dataset/splits/pascal/val.txt"

# Criar pastas se necessário
os.makedirs(split_base, exist_ok=True)
os.makedirs(os.path.dirname(val_path), exist_ok=True)

# Separar imagens rotuladas e não rotuladas
rotuladas = []
nao_rotuladas = []

for fname in os.listdir(image_dir):
    if not fname.lower().endswith(".jpg"):
        continue
    mask_name = fname.replace(".jpg", ".png")
    if os.path.exists(os.path.join(mask_dir, mask_name)):
        rotuladas.append(fname)
    else:
        nao_rotuladas.append(fname)

# Embaralhar
random.shuffle(rotuladas)
random.shuffle(nao_rotuladas)

# Divisão: 20% labeled, 20% val
num_total = len(rotuladas)
num_labeled = int((1/16) * num_total) #num_labeled = int(0.2 * num_total)
num_val = int(0.1 * num_total)

labeled_set = rotuladas[:num_labeled]
val_set = rotuladas[num_labeled:num_labeled + num_val]
# O restante das rotuladas será ignorado neste script

def write_split(file_path, file_list, include_mask=True):
    with open(file_path, "w") as f:
        for filename in file_list:
            img_path = f"pascal/JPEGImages/{filename}"
            if include_mask:
                mask_path = f"pascal/SegmentationClass/{filename.replace('.jpg', '.png')}"
                f.write(f"{img_path} {mask_path}\n")
            else:
                f.write(f"{img_path}\n")

# Escrever arquivos
write_split(os.path.join(split_base, "labeled.txt"), labeled_set, include_mask=True)
write_split(os.path.join(split_base, "unlabeled.txt"), nao_rotuladas, include_mask=False)
write_split(val_path, val_set, include_mask=True)

# Contagem
print("✅ Arquivos gerados:")
print(f"  - labeled.txt ({len(labeled_set)})")
print(f"  - val.txt ({len(val_set)})")
print(f"  - unlabeled.txt ({len(nao_rotuladas)})")
print(f"Total de imagens rotuladas (detectadas): {len(rotuladas)}")
print(f"Total de imagens não rotuladas: {len(nao_rotuladas)}")
print(f"Total de imagens em JPEGImages: {len(rotuladas) + len(nao_rotuladas)}")
