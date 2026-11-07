[CmdletBinding()]
param(
    [string[]]$Dataset = @("all"),
    [string]$Manifest = (Join-Path $PSScriptRoot "..\data\source\jangafx\manifests\sources.json"),
    [string]$ArchiveDir = (Join-Path $PSScriptRoot "..\data\source\jangafx\archives"),
    [string]$VbtMetadataRoot = (Join-Path $PSScriptRoot "..\data\smoke")
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

$sevenZip = @(
    "C:\ProgramData\chocolatey\tools\7z.exe",
    "C:\Program Files\7-Zip\7z.exe",
    "C:\Program Files\NVIDIA Corporation\NVIDIA App\7z.exe"
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $sevenZip) {
    $sevenZip = (Get-Command 7z.exe -ErrorAction Stop).Source
}

$rows = foreach ($item in $selected) {
    $archivePath = Join-Path $ArchiveDir $item.archive_name
    if (-not (Test-Path -LiteralPath $archivePath)) {
        throw "Archive is missing: $archivePath"
    }
    $archive = Get-Item -LiteralPath $archivePath
    if ($archive.Length -ne [long]$item.archive_bytes) {
        throw "Archive size mismatch for $($item.id): expected=$($item.archive_bytes) actual=$($archive.Length)"
    }

    $listing = (& $sevenZip l -slt $archive.FullName 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) {
        throw "7-Zip listing failed for $($item.id): $listing"
    }
    $entries = @($listing -split '(?:\r?\n){2,}' | ForEach-Object {
        $pathMatch = [regex]::Match($_, '(?m)^Path = (.+?)\r?$')
        $sizeMatch = [regex]::Match($_, '(?m)^Size = (\d+)\r?$')
        if ($pathMatch.Success -and $sizeMatch.Success) {
            [pscustomobject]@{
                path = $pathMatch.Groups[1].Value.Trim()
                name = [System.IO.Path]::GetFileName($pathMatch.Groups[1].Value.Trim())
                extension = [System.IO.Path]::GetExtension($pathMatch.Groups[1].Value.Trim()).ToLowerInvariant()
                bytes = [long]$sizeMatch.Groups[1].Value
            }
        }
    })
    $vdbEntries = @($entries | Where-Object extension -eq ".vdb")
    $mediaEntries = @($entries | Where-Object { $_.extension -in ".mp4", ".webm", ".mov", ".avi" })
    $presetEntries = @($entries | Where-Object extension -eq ".ember")

    $metadataPath = Join-Path (Join-Path $VbtMetadataRoot $item.id) ($item.id + "_density.metadata.json")
    if (-not (Test-Path -LiteralPath $metadataPath)) {
        throw "Canonical VBT metadata is missing: $metadataPath"
    }
    $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    $expectedNames = @{}
    foreach ($name in $metadata.frame_files) { $expectedNames[[string]$name] = $true }
    $matchedEntries = @($vdbEntries | Where-Object { $expectedNames.ContainsKey($_.name) })
    if ($matchedEntries.Count -ne $expectedNames.Count) {
        $foundNames = @{}
        foreach ($entry in $matchedEntries) { $foundNames[$entry.name] = $true }
        $missing = @($expectedNames.Keys | Where-Object { -not $foundNames.ContainsKey($_) })
        throw "Archive/VBT frame mismatch for $($item.id): expected=$($expectedNames.Count) found=$($matchedEntries.Count) missing=$($missing -join ', ')"
    }

    [pscustomobject]@{
        dataset = $item.id
        official_name = $item.official_name
        archive = $archive.Name
        archive_bytes = $archive.Length
        page_advertised_frames = [int]$item.page_advertised_frames
        archive_vdb_frames = $vdbEntries.Count
        frame_count_matches_page = ($vdbEntries.Count -eq [int]$item.page_advertised_frames)
        archive_vdb_bytes = [long](($vdbEntries | Measure-Object bytes -Sum).Sum)
        vbt_matched_frames = $matchedEntries.Count
        matched_source_vdb_bytes = [long](($matchedEntries | Measure-Object bytes -Sum).Sum)
        extra_archive_frames = $vdbEntries.Count - $matchedEntries.Count
        preview_files = $mediaEntries.Count
        preview_unpacked_bytes = [long](($mediaEntries | Measure-Object bytes -Sum).Sum)
        preset_files = $presetEntries.Count
    }
}

$inventoryPath = Join-Path (Split-Path -Parent $manifestPath) "archive_inventory.csv"
$rows | Export-Csv -LiteralPath $inventoryPath -NoTypeInformation -Encoding UTF8
$rows | Format-Table -AutoSize
Write-Host "Wrote $inventoryPath"
