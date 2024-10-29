# 🍽 FoodVision Mini

**Arpit Vaghela**  
*Computer Engineering Student at the University of Guelph*

---

## 🚀 Introducing FoodVision Mini: A Vision Transformer-Based Food Classifier

Over the past three months, I've been working on an exciting computer vision project called **FoodVision Mini**. This model classifies images of food into three categories: 🍕 Pizza, 🥩 Steak, and 🍣 Sushi, using the **Vision Transformer (ViT) B16** model.

## 🔍 What is a Vision Transformer?

Vision Transformers (ViTs) adapt the Transformer model, traditionally used in natural language processing, for image recognition. Unlike CNNs, which use local connections to capture patterns, ViTs use an **attention mechanism** that learns to focus on the most important parts of an image. This model divides an image into small patches (16x16 pixels for ViT B16), treating each patch like a word in a sentence, allowing it to recognize global patterns and dependencies effectively.

This shift enables ViTs to perform well on large datasets and complex image classification tasks, making them a powerful alternative to CNNs.

## 📝 Inspiration for FoodVision Mini

This project is inspired by the research paper titled ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929). The paper introduces Vision Transformers and demonstrates their potential to match or outperform traditional CNNs, especially on larger datasets. FoodVision Mini leverages this innovative approach to achieve impressive accuracy in classifying pizza, steak, and sushi.

## 🔑 What is Vision Transformer B16?

The **Vision Transformer B16** is a model where:
- **B**: Indicates the base model size.
- **16**: Divides the input image into 16x16 patches, each treated as a token.

Using **PyTorch's torchvision implementation** of ViT B16, I applied transfer learning by fine-tuning the model with pre-trained weights from ImageNet, which helped in quickly adapting the model to my specific dataset.

🔗 [ViT B16 Documentation](https://pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html)

## 🔑 Why PyTorch?

I chose **PyTorch** for its flexibility, ease of experimentation, and strong community support. Its dynamic graphing simplifies tweaking and debugging, which was essential for fine-tuning the Vision Transformer B16. The combination of **PyTorch and torchvision** made it straightforward to utilize the ViT model.

## 🍕🥩🍣 Dataset

The **FoodVision Mini** dataset consists of three classes:
- **Pizza**: Varying toppings and crusts.
- **Steak**: Different cuts and cooking styles.
- **Sushi**: Wide variety, including nigiri and rolls.

The ViT B16’s ability to capture fine details and global patterns made it ideal for classifying this diverse dataset.

## 🌐 Development Environment: Google Colab

I used **Google Colab** for training and experimentation due to:
- Free GPU access.
- Ease of use and setup.
- Native support for PyTorch and TensorFlow.

## 🖥️ Bringing FoodVision Mini to Life: Gradio Demo

FoodVision Mini is accessible through an interactive demo built with **Gradio**. This tool provides a simple web interface, making it easy for anyone to interact with the model.

## ☺ What is Hugging Face Spaces?

I hosted the FoodVision Mini demo on **Hugging Face Spaces**, a platform for ML model sharing. It allows live model demos for a hands-on experience, making the project accessible to others.

## 🧪 Model Deployment: Experiment Summary

My goal was to deploy a model that:
- **Performs well**: Achieves 95%+ accuracy.
- **Runs quickly**: Low latency for real-time use (aiming for 30ms or faster predictions).

**ViT Model Summary**:
- **Total Parameters**: 85,800,963
- **Trainable Parameters**: 2,307
- **Test Accuracy**: 97.22%
- **Model Size**: 327 MB
- **Prediction Time (CPU)**: 0.4373 seconds

Using **CrossEntropyLoss** and the **Adam optimizer** (learning rate = 1e-3), the model reached **97.2% test accuracy** and **0.0667 test loss** after 20 epochs.

---

## 💻 Tech Stack

- **PyTorch**: Deep learning framework.
- **torchvision**: Pre-trained Vision Transformer B16.
- **Google Colab**: For training with GPU support.
- **Gradio**: Building an interactive web demo.
- **Hugging Face Spaces**: Hosting the live demo.

🔗 Check out the FoodVision Mini demo on Hugging Face Spaces: [link](https://huggingface.co/spaces/vapit/foodvision_mini)

I have been working for three months to read, research, and learn how to implement and develop this model! I hope you give it a fair try and see if it predicts correctly or not! If you want feedback, learn how I made it, or access my Google Colab notebook, feel free to reach out to me.
