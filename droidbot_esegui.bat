@ECHO OFF
REM Eseguo DroidBot sul primo APK
droidbot -a "C:\Users\juve2\StudioProjects\PassAndroid\android\build\intermediates\apk\noMapsNoAnalyticsForFDroid\debug\PassAndroid-3.7.3-noMaps-noAnalytics-forFDroid-debug.apk" -o output-pass-llama -is_emulator -accessibility_auto -timeout 10800

REM Eseguo DroidBot sul secondo APK
droidbot -a "C:\Users\juve2\StudioProjects\Omni-Notes\omniNotes\build\intermediates\apk\alpha\debug\OmniNotes-alphaDebug-6.4.0.apk" -o output-note-llama -is_emulator -accessibility_auto -timeout 10800

REM Eseguo DroidBot sul terzo APK
droidbot -a "C:\Users\juve2\StudioProjects\thunderbird-android\app-k9mail\build\outputs\apk\foss\debug\app-k9mail-foss-debug.apk" -o output-tfa-llama -is_emulator -accessibility_auto -timeout 10800

PAUSE
