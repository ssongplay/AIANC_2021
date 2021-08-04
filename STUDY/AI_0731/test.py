# 위상 반전, 파장 결합(Merge), 소리 재생 하는 소스코드

from pydub import AudioSegment
from pydub.playback import play

# 기존 wav 파일 이름 지정
original_FileName = "Beep.wav"
pred_FileName = "pred.wav"

# 지정한 y_val wav 파일 load
original_sound = AudioSegment.from_file(original_FileName, format="wav")

# 지정한 pred wav 파일 load
pred_sound = AudioSegment.from_file(pred_FileName, format="wav")

# 예측 파일 역위상으로 뒤집기
pred_sound_reverse = pred_sound.invert_phase()

# 역위상 파장 wav파일로 저장
pred_sound_reverse.export("pred_sound_reverse.wav", format="wav")



# 두개의 소리(wav파일)를 결합(Merge)
combinedSound_AI = original_sound.overlay(pred_sound_reverse)
# 결합된 소리를 저장
combinedSound_AI.export("combined_AI.wav", format="wav")

# Play audio file :
# should play nothing since two files with inverse phase cancel each other

# 기존 wav 파일 재생
print("Beep.wav")
play(original_sound)
# 위상 반전된 wav 파일 재생
print("예측한 소리의 역위상")
play(pred_sound_reverse)
# 결합된 wav 파일 재생
print("결합된 소리")
play(combinedSound_AI)
