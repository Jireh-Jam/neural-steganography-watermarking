def test_embed_extract(encoder, decoder, image_path, message):
    """Test the steganography system with proper shape handling"""
    try:
        # 1. Load and prepare cover image
        cover_img = load_image(image_path)
        if len(cover_img.shape) == 3:
            cover_img = np.expand_dims(cover_img, 0)  # Add batch dim if needed
        if len(cover_img.shape) != 4:
            raise ValueError(f"Image must be 4D (batch,h,w,c), got {cover_img.shape}")

        # 2. Prepare message tensor with EXACT SAME batch size
        message_arr = text_to_bits(message)
        if len(message_arr.shape) == 3:
            message_arr = np.expand_dims(message_arr, 0)
        
        # Ensure identical batch dimensions
        if cover_img.shape[0] != message_arr.shape[0]:
            # Repeat message to match image batch size
            message_arr = np.repeat(message_arr, cover_img.shape[0], axis=0)

        # 3. Verify shapes before prediction
        print(f"Input shapes - Cover: {cover_img.shape}, Message: {message_arr.shape}")
                
                # 1. Embedding Test
        stego_img = encoder.predict([cover_img, message_arr], verbose=0)
        if len(stego_img) == 0:
            raise RuntimeError("Encoder returned empty output")
        stego_img = stego_img[0]  # Get first batch element
        
        # 2. Extraction Test
        if len(stego_img.shape) == 3:
            stego_img = np.expand_dims(stego_img, 0)
        pred_msg = decoder.predict(stego_img, verbose=0)[0]    
            # Post-processing
        recovered_text = bits_to_text(pred_msg)
        
        # 3. Bit-level comparison
        original_bits = message_arr.flatten()
        decoded_bits = (pred_msg > 0.5).astype(int).flatten()
        
        # Trim to match lengths
        min_length = min(len(original_bits), len(decoded_bits))
        original_bits = original_bits[:min_length]
        decoded_bits = decoded_bits[:min_length]
        
        # Create comparison string
        bit_comparison = []
        for orig, dec in zip(original_bits, decoded_bits):
            if orig == dec:
                bit_comparison.append(f"{int(orig)}✓")
            else:
                bit_comparison.append(f"{int(orig)}→{int(dec)}✗")
        # 4. Embed and extract with shape validation
        stego_img = encoder.predict([cover_img, message_arr], verbose=1)[0]
        pred_msg = decoder.predict(np.expand_dims(stego_img, 0), verbose=1)[0]
        img_psnr = psnr(
            cover_img[0] if len(cover_img.shape) == 4 else cover_img,
            stego_img[0] if len(stego_img.shape) == 4 else stego_img
        )
        # 5. Process results
        recovered_text = bits_to_text(pred_msg)
        
        # 6. Visualization with shape checks
        visualize_stego_results(
            np.squeeze(cover_img),  # Remove batch dim for display
            np.squeeze(stego_img)
        )
        print("\n=== BIT-LEVEL COMPARISON ===")
        print(f"Original Bits: {''.join(map(str, original_bits.astype(int)))}")
        print(f"Decoded Bits:  {''.join(map(str, decoded_bits))}")
        print(f"Matching Bits: {' '.join(bit_comparison[:100])}" + 
              ("..." if len(bit_comparison) > 100 else ""))
        
        print("\n=== TEXT COMPARISON ===")
        print(f"Original Text: {message}")
        print(f"Decoded Text:  {recovered_text}")
        
        print("\n=== SUMMARY ===")
        print(f"Bit Accuracy: {np.mean(original_bits == decoded_bits):.2%}")
        print(f"Exact Match: {'✅' if message == recovered_text else '❌'}")
        print(f"PSNR: {img_psnr:.2f} dB")
        return {
            'original_image': cover_img,
            'stego_image': stego_img,
            'psnr': img_psnr,
            'recovered_message': recovered_text
        }

    except Exception as e:
        print(f"Test failed: {str(e)}")
        traceback.print_exc()
        return None
