Set oShell = CreateObject ("Wscript.Shell") 
Dim strArgs
strArgs = "cmd /c rt/callc.bat"
oShell.Run strArgs, 0, false