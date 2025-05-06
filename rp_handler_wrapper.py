from rp_handler import handler as real_handler

print("[DEBUG] Wrapper module loaded.")


def handler(event):
    print("[DEBUG] Wrapper handler invoked")
    return real_handler(event)
