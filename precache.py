import whisperx

def main():
    # 1) Whisper model (one model works for all languages)
    whisperx.load_model("large-v3", "cpu", compute_type="int8", language=None)
    print("✅ Cached Whisper large-v3")

    # 2) Alignment models (language-dependent)
    langs = [
        "en","es","fr","de","it","pt","nl","pl","cs","sk","sl","hr","sr","bs","hu","ro","bg","ru","uk","el","tr",
        "ar","he","fa","ur","hi","bn","ta","te","kn","ml","mr","gu","pa",
        "zh","ja","ko","th","vi","id","ms","tl",
        "sv","no","da","fi","is",
        "et","lv","lt"
    ]

    ok, fail = 0, 0
    for l in langs:
        try:
            whisperx.load_align_model(language_code=l, device="cpu")
            ok += 1
            print(f"✅ Cached align: {l}")
        except Exception as e:
            fail += 1
            print(f"⚠️ Skip align {l}: {e}")

    print(f"Done. align ok={ok} fail={fail}")

if __name__ == "__main__":
    main()
