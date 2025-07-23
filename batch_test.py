"""
batch_test.py - Comprehensive evaluation of neural steganography system
"""

import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from models import encoder, decoder
from utils import load_image, text_to_bits, bits_to_text

def batch_test(image_paths, secret_message, max_images=None):
    """
    Batch test steganography system on multiple images
    
    Args:
        image_paths: List of paths to test images
        secret_message: Secret text to embed
        max_images: Maximum number of images to process (None for all)
    
    Returns:
        Dictionary containing aggregated metrics and per-image results
    """
    # Initialize metrics
    results = {
        'psnr': [],
        'ssim': [],
        'bit_accuracy': [],
        'exact_matches': 0,
        'failed': 0,
        'per_image': []
    }
    
    # Limit number of images if specified
    if max_images:
        image_paths = image_paths[:max_images]
    
    print(f"\nTesting on {len(image_paths)} images with message: '{secret_message}'")
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Test current image
            test_result = test_embed_extract(
                encoder=encoder,
                decoder=decoder,
                image_path=img_path,
                message=secret_message
            )
            
            if test_result is None:
                results['failed'] += 1
                continue
            
            # Calculate additional metrics
            original = test_result['original_image']
            stego = test_result['stego_image']
            
            # Remove batch dimension if present
            if len(original.shape) == 4:
                original = original[0]
            if len(stego.shape) == 4:
                stego = stego[0]
            
            # Compute metrics
            current_ssim = ssim(original, stego, multichannel=True, 
                               data_range=stego.max() - stego.min())
            original_bits = text_to_bits(secret_message).flatten()
            decoded_bits = (text_to_bits(test_result['recovered_message'])).flatten()
            
            # Trim to same length
            min_len = min(len(original_bits), len(decoded_bits))
            bit_accuracy = np.mean(original_bits[:min_len] == decoded_bits[:min_len])
            
            # Update results
            results['psnr'].append(test_result['psnr'])
            results['ssim'].append(current_ssim)
            results['bit_accuracy'].append(bit_accuracy)
            
            if secret_message == test_result['recovered_message']:
                results['exact_matches'] += 1
                
            # Store per-image results
            results['per_image'].append({
                'path': img_path,
                'psnr': test_result['psnr'],
                'ssim': current_ssim,
                'bit_accuracy': bit_accuracy,
                'exact_match': secret_message == test_result['recovered_message'],
                'recovered_message': test_result['recovered_message']
            })
            
        except Exception as e:
            print(f"\nError processing {img_path}: {str(e)}")
            results['failed'] += 1
    
    # Calculate aggregate statistics
    if results['psnr']:  # Only if we have successful tests
        results['avg_psnr'] = np.mean(results['psnr'])
        results['avg_ssim'] = np.mean(results['ssim'])
        results['avg_bit_accuracy'] = np.mean(results['bit_accuracy'])
        results['success_rate'] = len(results['psnr']) / len(image_paths)
    else:
        results.update({
            'avg_psnr': 0,
            'avg_ssim': 0,
            'avg_bit_accuracy': 0,
            'success_rate': 0
        })
    
    return results

def print_summary(results):
    """Print formatted test summary"""
    print("\n=== TEST SUMMARY ===")
    print(f"Images Processed: {len(results['per_image']) + results['failed']}")
    print(f"Successful Tests: {len(results['per_image'])}")
    print(f"Failed Tests: {results['failed']}")
    print(f"\nQuality Metrics (avg):")
    print(f"PSNR: {results['avg_psnr']:.2f} dB")
    print(f"SSIM: {results['avg_ssim']:.4f}")
    print(f"Bit Accuracy: {results['avg_bit_accuracy']:.2%}")
    print(f"\nExact Message Matches: {results['exact_matches']}/{len(results['per_image'])}")
    print(f"Success Rate: {results['success_rate']:.2%}")

if __name__ == "__main__":
    # Example usage
    import glob
    
    # 1. Load test images
    test_images = glob.glob("data/test/*.jpg")[:100]  # First 100 test images
    
    # 2. Define secret message
    secret_msg = "MySecretKey123"
    
    # 3. Run batch test
    test_results = batch_test(
        image_paths=test_images,
        secret_message=secret_msg,
        max_images=None  # Process all images
    )
    
    # 4. Print summary
    print_summary(test_results)
    
    # 5. Save detailed results
    import pandas as pd
    df = pd.DataFrame(test_results['per_image'])
    df.to_csv("steganography_test_results.csv", index=False)
    print("\nDetailed results saved to steganography_test_results.csv")
