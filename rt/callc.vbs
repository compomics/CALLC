Set oShell = CreateObject ("Wscript.Shell") 
Dim strArgs
strArgs = "cmd /c callc.bat"
oShell.Run strArgs, 0, false