from model import HybridModel, train_model, freeze, train_only, unfreeze_all, evaluate, load_pretrained, get_cifar10_loader

if __name__ == "__main__":
    # Test individual model accuracies
    print("Loading EfficientNet...")
    eff_path = 'efficientnet.pth'  # Replace with actual path if needed
    eff = load_pretrained(eff_path)
    # Use smaller batch size to fit on limited-GPU memory
    loader = get_cifar10_loader(train=True, download=True, batch_size=32)
    acc_eff = evaluate(eff, loader)
    print(f"EfficientNet accuracy: {acc_eff:.4f}")

    print("Loading BIASNN...")
    bias_path = 'biasnn.pth'  # Replace with actual path
    bias = load_pretrained(bias_path)
    acc_bias = evaluate(bias, loader)
    print(f"BIASNN accuracy: {acc_bias:.4f}")

    # Hybrid: first half EfficientNet, second half BIASNN
    cut_point = len(list(eff.children())) // 2
    print(f"Cut point: {cut_point}")

    print("Creating hybrid model...")
    hybrid = HybridModel(eff_path, bias_path, cut_point)

    print("Freezing encoder and decoder...")
    freeze(hybrid.encoder)
    freeze(hybrid.decoder)

    print("Training only the adapter...")
    if hybrid.adapter:
        train_only(hybrid, hybrid.adapter)
    train_model(hybrid, epochs=1, dataloader=loader)

    print("Evaluating hybrid model after adapter training...")
    acc_hybrid = evaluate(hybrid, loader)
    print(f"Hybrid model accuracy: {acc_hybrid:.4f}")

    print("Hybrid model training complete.")
