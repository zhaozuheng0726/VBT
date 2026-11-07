[CmdletBinding()]
param(
    [string[]]$Dataset = @("all"),
    [string]$Manifest = (Join-Path $PSScriptRoot "..\data\source\jangafx\manifests\sources.json"),
    [string]$ArchiveDir = (Join-Path $PSScriptRoot "..\data\source\jangafx\archives")
)

$ErrorActionPreference = "Stop"
$manifestPath = (Resolve-Path -LiteralPath $Manifest).Path
$catalog = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$selected = if ($Dataset -contains "all") {
    @($catalog.datasets)
} else {
    @($catalog.datasets | Where-Object { $Dataset -contains $_.id })
}

if ($selected.Count -eq 0) {
    throw "No datasets matched: $($Dataset -join ', ')"
}

New-Item -ItemType Directory -Force -Path $ArchiveDir | Out-Null
$curl = if (Test-Path -LiteralPath "$env:SystemRoot\System32\curl.exe") {
    "$env:SystemRoot\System32\curl.exe"
} else {
    (Get-Command curl.exe -ErrorAction Stop).Source
}
$aria = Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot "..\third_party\aria2_portable") -Recurse -File -Filter "aria2c.exe" -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty FullName

function Resolve-MediaFireDownload {
    param([Parameter(Mandatory)][string]$PageUrl)

    $html = (Invoke-WebRequest -UseBasicParsing -Uri $PageUrl -TimeoutSec 90).Content
    $match = [regex]::Match($html, 'https://download[^"''< ]+')
    if (-not $match.Success) {
        throw "MediaFire direct link was not found: $PageUrl"
    }
    return $match.Value.Replace('&amp;', '&')
}

$results = foreach ($item in $selected) {
    $output = Join-Path $ArchiveDir $item.archive_name
    $expected = [long]$item.archive_bytes
    $attempt = 0

    while ((-not (Test-Path -LiteralPath $output)) -or
           ((Get-Item -LiteralPath $output).Length -ne $expected) -or
           (Test-Path -LiteralPath ($output + ".aria2"))) {
        $attempt++
        if ($attempt -gt 8) {
            throw "Download failed size validation after 8 attempts: $($item.id)"
        }

        $directUrl = Resolve-MediaFireDownload -PageUrl $item.mediafire_url
        $head = Invoke-WebRequest -UseBasicParsing -Method Head -Uri $directUrl -TimeoutSec 90
        $remoteBytes = [long]$head.Headers.'Content-Length'
        if ($remoteBytes -ne $expected) {
            throw "Remote size changed for $($item.id): manifest=$expected remote=$remoteBytes"
        }

        Write-Host "[$($item.id)] downloading $($item.archive_name) ($remoteBytes bytes)"
        if ($aria) {
            $outputDirectory = Split-Path -Parent $output
            $outputName = Split-Path -Leaf $output
            & $aria --continue=true --max-connection-per-server=8 --split=8 --min-split-size=4M --file-allocation=none --auto-file-renaming=false --allow-overwrite=true --max-tries=3 --retry-wait=5 --connect-timeout=30 --timeout=60 --console-log-level=warn --summary-interval=0 "--dir=$outputDirectory" "--out=$outputName" $directUrl
        } else {
            & $curl --silent --show-error -L --fail --retry 5 --retry-delay 5 --connect-timeout 30 --speed-time 60 --speed-limit 131072 -C - --output $output $directUrl
        }
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Downloader exited with $LASTEXITCODE; refreshing the temporary link"
        }
    }

    $file = Get-Item -LiteralPath $output
    $hash = (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash
    [pscustomobject]@{
        dataset = $item.id
        archive = $file.Name
        bytes = $file.Length
        sha256 = $hash
        source_url = $item.mediafire_url
    }
}

$resultPath = Join-Path (Split-Path -Parent $manifestPath) "downloaded_archives.csv"
$results | Export-Csv -LiteralPath $resultPath -NoTypeInformation -Encoding UTF8
$results | Format-Table -AutoSize
Write-Host "Wrote $resultPath"
