# Install Spell Checker for OCR Improvements
# This script installs the pyspellchecker library for automatic spell correction

Write-Host "Installing pyspellchecker for OCR spell correction..." -ForegroundColor Cyan

try {
    # Install pyspellchecker
    pip install pyspellchecker
    
    Write-Host "`n✓ Successfully installed pyspellchecker!" -ForegroundColor Green
    Write-Host "`nThe OCR pipeline will now automatically correct spelling errors." -ForegroundColor Green
    Write-Host "Restart your Flask application to use the new feature." -ForegroundColor Yellow
    
} catch {
    Write-Host "`n✗ Installation failed: $_" -ForegroundColor Red
    Write-Host "`nTry running: pip install pyspellchecker" -ForegroundColor Yellow
    exit 1
}

Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

