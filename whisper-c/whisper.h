#include <filesystem>

typedef struct Whisper Whisper;

Whisper load(std::filesystem::path const& modelPath);