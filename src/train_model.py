import os
import json
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns  # si pas installé, enlever la partie matrice de confusion ou pip install seaborn
print(">>> SCRIPT train_model.py LANCÉ")


# =========================
# 1. Paramètres globaux
# =========================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 15
EPOCHS_FINETUNE = 10

train_dir = os.path.join("data", "train")
test_dir = os.path.join("data", "test")


def create_generators():
    """Crée les générateurs train/val/test avec augmentation et normalisation."""
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,  # 80% train / 20% val
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # Sauvegarder le mapping classe -> index pour l'app Streamlit
    class_indices = train_generator.class_indices
    with open("class_indices.json", "w") as f:
        json.dump(class_indices, f, indent=4)

    print("Classes trouvées :", class_indices)

    return train_generator, val_generator, test_generator


def build_model(num_classes):
    """Construit le modèle EfficientNetB0 + tête de classification."""
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    # On freeze le backbone au début
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model


def train_head(model, train_gen, val_gen):
    """Entraînement de la tête de classification (backbone figé)."""
    checkpoint = ModelCheckpoint(
        "best_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    )

    history = model.fit(
        train_gen,
        epochs=EPOCHS_HEAD,
        validation_data=val_gen,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    return history


def fine_tune(model, base_model, train_gen, val_gen):
    """Fine-tuning : on défige une partie du backbone pour améliorer les performances."""
    base_model.trainable = True

    # On freeze ~2/3 des premières couches, on entraine le reste
    fine_tune_at = int(len(base_model.layers) * 0.66)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint = ModelCheckpoint(
        "best_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    )

    history_ft = model.fit(
        train_gen,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_gen,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    return history_ft


def plot_history(history, title_prefix=""):
    """Affiche les courbes loss/accuracy (train/val) pour le rapport."""
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    epochs_range = range(1, len(acc) + 1)

    plt.figure()
    plt.plot(epochs_range, acc, label='Train acc')
    plt.plot(epochs_range, val_acc, label='Val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{title_prefix} Accuracy')
    plt.legend()
    plt.savefig(f"{title_prefix.lower().replace(' ', '_')}_accuracy.png")
    plt.close()

    plt.figure()
    plt.plot(epochs_range, loss, label='Train loss')
    plt.plot(epochs_range, val_loss, label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title_prefix} Loss')
    plt.legend()
    plt.savefig(f"{title_prefix.lower().replace(' ', '_')}_loss.png")
    plt.close()


def evaluate_model(test_gen):
    """Charge le meilleur modèle et calcule les métriques + matrice de confusion."""
    model = load_model("best_model.h5")

    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

    y_pred = model.predict(test_gen)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true = test_gen.classes

    class_indices = test_gen.class_indices
    idx_to_class = {v: k for k, v in class_indices.items()}
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    print("\nClassification report :")
    print(classification_report(y_true, y_pred_labels, target_names=target_names))

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()


def main():
    # 1. Générateurs
    train_gen, val_gen, test_gen = create_generators()
    num_classes = train_gen.num_classes

    # 2. Construction du modèle
    model, base_model = build_model(num_classes)
    model.summary()

    # 3. Entraînement de la tête
    history = train_head(model, train_gen, val_gen)
    plot_history(history, title_prefix="Head Training")

    # 4. Fine-tuning
    history_ft = fine_tune(model, base_model, train_gen, val_gen)
    plot_history(history_ft, title_prefix="Fine Tuning")

    # 5. Évaluation finale sur le test
    evaluate_model(test_gen)
    print("Entraînement et évaluation terminés. Modèle sauvegardé dans best_model.h5")


if __name__ == "__main__":
    print(">>> AVANT main()")
    main()
    print(">>> APRES main()")

