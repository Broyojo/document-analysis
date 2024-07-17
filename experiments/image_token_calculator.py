def calculate_image_tokens(width, height, detail="high"):
    if detail == "low":
        return 85, width, height

    # Scale down to fit within 2048x2048 square
    if width > 2048 or height > 2048:
        aspect_ratio = width / height
        if width > height:
            width = 2048
            height = int(width / aspect_ratio)
        else:
            height = 2048
            width = int(height * aspect_ratio)

    # Scale such that the shortest side is 768px
    if width < height:
        if width != 768:
            scale_factor = 768 / width
            width = 768
            height = int(height * scale_factor)
    else:
        if height != 768:
            scale_factor = 768 / height
            height = 768
            width = int(width * scale_factor)

    # Calculate number of 512px tiles
    tiles_x = (width + 511) // 512  # Ceiling division
    tiles_y = (height + 511) // 512
    total_tiles = tiles_x * tiles_y

    # Calculate token cost
    tokens = (170 * total_tiles) + 85

    return tokens, width, height
