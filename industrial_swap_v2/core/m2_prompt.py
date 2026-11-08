"""CPU-only prompt construction shared by Web preview and M2 execution."""
def build_prompt(ctx):
    override = ctx.cfg['m2'].get('prompt_override')
    if override is not None:
        if not isinstance(override, str) or not override.strip():
            raise ValueError('Custom generation prompt must not be empty')
        return override.strip()
    if ctx.cfg['m2'].get('prompt_mode')=='generic_device':
        return (
            "Edit image 1 and return one photorealistic RGB image of the same factory from exactly the same camera. "
            "Image 1 is the original scene. Image 2 is ONLY a position guide: the translucent magenta mask marks the visible old device and its allowed local replacement space. Its boundary may be irregular and may contain protected holes for foreground equipment. Remove all magenta markings in the output. "
            "Replace the entire selected old device in the marked region: " + ctx.cfg['m1']['target_text'] + ". "
            "Image 3 is ONLY a background depth reference. The editing area is blacked out as UNKNOWN depth, not a surface to copy. Use the surrounding depth for perspective and context, not the old device shape or grayscale appearance. "
            "Generate exactly one complete replacement device: " + ctx.cfg['m2']['replacement_text'] + ". "
            "Place its support base or wheels on the floor at the old device location. Fit the complete device inside the marked volume without clipping, floating parts or intersections with neighboring equipment. "
            "Preserve every unmarked area, including foreground objects and holes inside the marked region. Keep their real occlusion in front of the new device. Preserve the original camera, surrounding equipment, containers, railings, floor markings and lighting. Match scene perspective, scale, color temperature and shading. "
            "Do not add extra devices, people, text, arrows, annotations or a new background. Return the edited full factory view, not a collage, isolated product photo or depth map."
        )
    return (
        "Edit image 1 and return one photorealistic RGB image of the same factory from exactly the same camera. "
        "Image 1 is the original scene. Image 2 is ONLY a position guide: the translucent magenta mask marks the allowed local editing region; preserve unmarked regions and foreground holes inside it. Remove all magenta markings in the output. "
        "Replace the entire old machine consisting of a black assembly and a white support base inside that polygon. "
        "Image 3 is ONLY a background depth reference, with the editing area blacked out as UNKNOWN depth. Use it for surrounding floor/equipment perspective, not the old machine's shape. Do not copy its grayscale appearance. "
        "Generate exactly one complete device: " + ctx.cfg["m2"]["replacement_text"] + " "
        "Show a coherent complete base, shoulder, elbow, wrist, distinct rotary joints and gripper with clear solid mechanical geometry. "
        "Place its base on the factory floor at the old device's base location. Fit the complete arm, gripper and pedestal within the marked volume without clipping. "
        "Keep the original camera, surrounding equipment, blue containers, railings, floor and lighting unchanged. Match the factory perspective, scale, color temperature and shading. "
        "Generate no cabinet, enclosure, extra machinery, duplicate arm, text, annotations or new background behind the robot. The output is the edited full factory view, not a collage or depth map."
    )
