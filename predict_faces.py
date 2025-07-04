"""
# Face Recognition Inference

This notebook demonstrates how to use the trained face recognition model to make predictions on new images.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import os
import pickle
from PIL import Image
import argparse

"""## 1. Load Trained Model"""

def load_face_recognition_model():
    """Load the trained face recognition model and label encoder"""
    print("🔄 Loading face recognition model...")

    # Load model
    model = load_model('models/best_face_model.h5')

    # Load label encoder
    with open('models/enhanced_label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    print(f"✅ Model loaded successfully!")
    print(f"📊 Model can recognize {len(label_encoder.classes_)} people")

    return model, label_encoder

"""## 2. Image Preprocessing"""

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess a single image for prediction

    Args:
        image_path: Path to the image file
        target_size: Target size for the model (default: 224x224)

    Returns:
        Preprocessed image array
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to target size
        img = cv2.resize(img, target_size)

        # Normalize pixel values
        img = img / 255.0

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        return img

    except Exception as e:
        print(f"❌ Error preprocessing image {image_path}: {e}")
        return None

"""## 3. Make Predictions"""

def predict_face(model, label_encoder, image_path, confidence_threshold=0.5):
    """
    Predict the identity of a face in an image

    Args:
        model: Trained face recognition model
        label_encoder: Label encoder for class names
        image_path: Path to the image file
        confidence_threshold: Minimum confidence to make a prediction

    Returns:
        Dictionary with prediction results
    """
    # Preprocess image
    img = preprocess_image(image_path)
    if img is None:
        return None

    # Make prediction
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # Get class name
    predicted_name = label_encoder.classes_[predicted_class]

    # Check confidence threshold
    if confidence < confidence_threshold:
        result = {
            'predicted_name': 'Unknown',
            'confidence': confidence,
            'all_predictions': predictions[0],
            'status': 'low_confidence'
        }
    else:
        result = {
            'predicted_name': predicted_name,
            'confidence': confidence,
            'all_predictions': predictions[0],
            'status': 'success'
        }

    return result

"""## 4. Visualize Predictions"""

def visualize_prediction(image_path, prediction_result, save_path=None):
    """
    Visualize the prediction result on the image

    Args:
        image_path: Path to the original image
        prediction_result: Prediction result dictionary
        save_path: Optional path to save the visualization
    """
    # Load original image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Show image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Input Image', fontsize=14, fontweight='bold')
    plt.axis('off')

    # Show prediction results
    plt.subplot(1, 2, 2)

    if prediction_result['status'] == 'success':
        # Show top predictions
        top_indices = np.argsort(prediction_result['all_predictions'])[-5:][::-1]
        top_names = [label_encoder.classes_[i] for i in top_indices]
        top_confidences = [prediction_result['all_predictions'][i] for i in top_indices]

        colors = ['green' if i == 0 else 'gray' for i in range(len(top_names))]
        bars = plt.barh(range(len(top_names)), top_confidences, color=colors, alpha=0.7)

        plt.yticks(range(len(top_names)), top_names)
        plt.xlabel('Confidence Score')
        plt.title(f'Top 5 Predictions\nPredicted: {prediction_result["predicted_name"]}',
                 fontsize=14, fontweight='bold')

        # Add confidence values on bars
        for i, (bar, conf) in enumerate(zip(bars, top_confidences)):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{conf:.3f}', va='center', fontweight='bold')

    else:
        plt.text(0.5, 0.5, f'Low Confidence\n({prediction_result["confidence"]:.3f})',
                ha='center', va='center', fontsize=16, color='red',
                transform=plt.gca().transAxes)
        plt.title('Prediction Result', fontsize=14, fontweight='bold')
        plt.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Visualization saved to {save_path}")

    plt.show()

"""## 5. Batch Prediction"""

def predict_batch(model, label_encoder, image_folder, confidence_threshold=0.5):
    """
    Make predictions on all images in a folder

    Args:
        model: Trained face recognition model
        label_encoder: Label encoder for class names
        image_folder: Path to folder containing images
        confidence_threshold: Minimum confidence threshold

    Returns:
        List of prediction results
    """
    results = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    print(f"🔄 Processing images in {image_folder}...")

    for filename in os.listdir(image_folder):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(image_folder, filename)

            print(f"📸 Processing {filename}...")
            prediction = predict_face(model, label_encoder, image_path, confidence_threshold)

            if prediction:
                prediction['image_path'] = image_path
                prediction['filename'] = filename
                results.append(prediction)

                # Print result
                if prediction['status'] == 'success':
                    print(f"   ✅ {prediction['predicted_name']} (confidence: {prediction['confidence']:.3f})")
                else:
                    print(f"   ⚠️ Unknown (confidence: {prediction['confidence']:.3f})")

    return results

"""## 6. Main Function"""

def main():
    """Main function for face recognition inference"""

    # Load model
    model, label_encoder = load_face_recognition_model()

    print("\n" + "="*60)
    print("🤖 FACE RECOGNITION INFERENCE")
    print("="*60)
    print("This script can:")
    print("1. Predict identity of a single image")
    print("2. Process all images in a folder")
    print("3. Visualize predictions with confidence scores")
    print("="*60)

    # Example usage
    print("\n📝 Example Usage:")
    print("python predict_faces.py --image path/to/image.jpg")
    print("python predict_faces.py --folder path/to/images/")
    print("python predict_faces.py --interactive")

    return model, label_encoder

"""## 7. Interactive Mode"""

def interactive_mode(model, label_encoder):
    """Interactive mode for testing predictions"""
    print("\n🎯 Interactive Mode")
    print("Enter image paths to test (or 'quit' to exit):")

    while True:
        image_path = input("\n📸 Enter image path: ").strip()

        if image_path.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break

        if not os.path.exists(image_path):
            print("❌ File not found. Please try again.")
            continue

        # Make prediction
        prediction = predict_face(model, label_encoder, image_path)

        if prediction:
            if prediction['status'] == 'success':
                print(f"✅ Predicted: {prediction['predicted_name']}")
                print(f"📊 Confidence: {prediction['confidence']:.3f}")
            else:
                print(f"⚠️ Unknown person (confidence: {prediction['confidence']:.3f})")

            # Ask if user wants to visualize
            visualize = input("🎨 Show visualization? (y/n): ").strip().lower()
            if visualize in ['y', 'yes']:
                visualize_prediction(image_path, prediction)

"""## 8. Command Line Interface"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Recognition Inference')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--val-random', type=int, help='Number of random validation images to predict (from X_val.npy)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--visualize', action='store_true', help='Show visualizations')

    args, unknown = parser.parse_known_args()

    # Load model
    model, label_encoder = load_face_recognition_model()

    if args.val_random:
        # Predict on N random validation images
        if not os.path.exists('X_val.npy') or not os.path.exists('y_val.npy'):
            print("❌ X_val.npy or y_val.npy not found. Please ensure you have a validation set saved.")
            exit(1)
        X_val = np.load('X_val.npy')
        y_val = np.load('y_val.npy')
        n = min(args.val_random, len(X_val))
        print(f"\n🔍 Predicting on {n} random images from validation set...")
        idxs = np.random.choice(len(X_val), n, replace=False)
        temp_files = []
        for i, idx in enumerate(idxs):
            img = (X_val[idx] * 255).astype(np.uint8) if X_val[idx].max() <= 1.0 else X_val[idx].astype(np.uint8)
            temp_path = f"val_random_{i}.jpg"
            cv2.imwrite(temp_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            temp_files.append(temp_path)
            prediction = predict_face(model, label_encoder, temp_path, args.confidence)
            actual_label = y_val[idx]
            actual_name = label_encoder.classes_[actual_label]
            if prediction:
                print(f"Image {i}: {temp_path}")
                print(f"   Actual:    {actual_name}")
                print(f"   Predicted: {prediction['predicted_name']} (confidence: {prediction['confidence']:.3f})")
                if args.visualize:
                    img_disp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    plt.imshow(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB))
                    plt.title(f'Actual: {actual_name}\nPredicted: {prediction["predicted_name"]} ({prediction["confidence"]:.3f})')
                    plt.axis('off')
                    plt.show()
        # Clean up temp files
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    elif args.image:
        # Single image prediction
        print(f"🎯 Predicting identity for {args.image}")
        prediction = predict_face(model, label_encoder, args.image, args.confidence)

        if prediction:
            if prediction['status'] == 'success':
                print(f"✅ Predicted: {prediction['predicted_name']}")
                print(f"📊 Confidence: {prediction['confidence']:.3f}")
            else:
                print(f"⚠️ Unknown person (confidence: {prediction['confidence']:.3f})")

            if args.visualize:
                visualize_prediction(args.image, prediction)

    elif args.interactive:
        # Interactive mode
        interactive_mode(model, label_encoder)

    else:
        # No arguments provided, show help
        main()
        print("\n💡 Use --help for command line options")
