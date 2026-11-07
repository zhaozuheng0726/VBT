[CmdletBinding()]
param(
    [string[]]$Dataset = @("all"),
    [string]$Manifest = (Join-Path $PSScriptRoot "..\data\source\jangafx\manifests\sources.json"),
    [string]$ArchiveDir = (Join-Path $PSScriptRoot "..\data\source\jangafx\archives"),
    [string]$OpenVdbRoot = (Join-Path $PSScriptRoot "..\data\source\jangafx\openvdb"),
    [string]$PreviewRoot = (Join-Path $PSScriptRoot "..\data\source\jangafx\previews"),
    [string]$PresetRoot = (Join-Path $PSScriptRoot "..\data\source\jangafx\presets"),
    [string]$VbtMetadataRoot = (Join-Path $PSScriptRoot "..\data\smoke"),
    [string]$VdbListTool = (Join-Path $PSScriptRoot "..\3D\vdb_tools\build\Release\vdb_to_raw.exe")
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
$listTool = (Resolve-Path -LiteralPath $VdbListTool).Path
New-Item -ItemType Directory -Force -Path $OpenVdbRoot, $PreviewRoot, $PresetRoot | Out-Null

$rows = foreach ($item in $selected) {
    $archivePath = Join-Path $ArchiveDir $item.archive_name
    if (-not (Test-Path -LiteralPath $archivePath)) {
        throw "Archive is missing: $archivePath"
    }
    $archive = Get-Item -LiteralPath $archivePath
    if ($archive.Length -ne [long]$item.archive_bytes) {
        throw "Archive size mismatch for $($item.id): expected=$($item.archive_bytes) actual=$($archive.Length)"
    }
    $archiveListing = (& $sevenZip l -slt $archive.FullName 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) {
        throw "7-Zip listing failed for $($item.id): $archiveListing"
    }
    $archiveVdbCount = [regex]::Matches($archiveListing, '(?im)^Path = .+\.vdb\r?$').Count
    if ($archiveVdbCount -eq 0) {
        throw "No VDB entries found in archive for $($item.id)"
    }

    $outputDir = Join-Path $OpenVdbRoot $item.id
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    $vdbFiles = @(Get-ChildItem -LiteralPath $outputDir -Recurse -File -Filter "*.vdb")
    if ($vdbFiles.Count -ne $archiveVdbCount) {
        Write-Host "[$($item.id)] extracting $($item.archive_name)"
        & $sevenZip x -y $archive.FullName ("-o" + $outputDir)
        if ($LASTEXITCODE -ne 0) {
            throw "7-Zip extraction failed for $($item.id) with exit code $LASTEXITCODE"
        }
        $vdbFiles = @(Get-ChildItem -LiteralPath $outputDir -Recurse -File -Filter "*.vdb")
    }
    if ($vdbFiles.Count -ne $archiveVdbCount) {
        throw "Extracted frame mismatch for $($item.id): archive=$archiveVdbCount extracted=$($vdbFiles.Count)"
    }
    if ($archiveVdbCount -ne [int]$item.page_advertised_frames) {
        Write-Warning "Archive frame count differs from the web page for $($item.id): page=$($item.page_advertised_frames) archive=$($vdbFiles.Count)"
    }

    $previewDir = Join-Path $PreviewRoot $item.id
    $presetDir = Join-Path $PresetRoot $item.id
    New-Item -ItemType Directory -Force -Path $previewDir, $presetDir | Out-Null
    $mediaFiles = @(Get-ChildItem -LiteralPath $outputDir -Recurse -File | Where-Object {
        $_.Extension -in ".mp4", ".webm", ".mov", ".avi"
    })
    foreach ($media in $mediaFiles) {
        Move-Item -LiteralPath $media.FullName -Destination (Join-Path $previewDir $media.Name) -Force
    }
    $presetFiles = @(Get-ChildItem -LiteralPath $outputDir -Recurse -File -Filter "*.ember")
    foreach ($preset in $presetFiles) {
        Move-Item -LiteralPath $preset.FullName -Destination (Join-Path $presetDir $preset.Name) -Force
    }

    $vdbFiles = @(Get-ChildItem -LiteralPath $outputDir -Recurse -File -Filter "*.vdb")
    $firstDir = $vdbFiles[0].DirectoryName
    $probeOutput = (& $listTool --input-dir $firstDir --output-prefix (Join-Path $env:TEMP ("jangafx_" + $item.id)) --grid density --list 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) {
        throw "VDB field probe failed for $($item.id): $probeOutput"
    }
    $fieldMatch = [regex]::Match($probeOutput, 'Float grids in first frame:\s*(.+)')
    $floatGrids = if ($fieldMatch.Success) { $fieldMatch.Groups[1].Value.Trim() } else { "unknown" }

    $metadataPath = Join-Path (Join-Path $VbtMetadataRoot $item.id) ($item.id + "_density.metadata.json")
    if (Test-Path -LiteralPath $metadataPath) {
        $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
        $frameNames = @{}
        foreach ($name in $metadata.frame_files) { $frameNames[[string]$name] = $true }
        $matchedFiles = @($vdbFiles | Where-Object { $frameNames.ContainsKey($_.Name) })
        if ($matchedFiles.Count -ne $frameNames.Count) {
            throw "VBT frame subset mismatch for $($item.id): expected=$($frameNames.Count) actual=$($matchedFiles.Count)"
        }
    } else {
        $matchedFiles = $vdbFiles
    }

    $previewFiles = @(Get-ChildItem -LiteralPath $previewDir -File)
    $presets = @(Get-ChildItem -LiteralPath $presetDir -File)
    [pscustomobject]@{
        dataset = $item.id
        official_name = $item.official_name
        archive = $archive.Name
        archive_bytes = $archive.Length
        vdb_frames_official = $vdbFiles.Count
        vdb_bytes_official = [long](($vdbFiles | Measure-Object Length -Sum).Sum)
        vbt_matched_frames = $matchedFiles.Count
        matched_source_vdb_bytes = [long](($matchedFiles | Measure-Object Length -Sum).Sum)
        first_frame_float_grids = $floatGrids
        preview_files = $previewFiles.Count
        preview_bytes = [long](($previewFiles | Measure-Object Length -Sum).Sum)
        preset_files = $presets.Count
    }
}

$inventoryPath = Join-Path (Split-Path -Parent $manifestPath) "extraction_inventory.csv"
$rows | Export-Csv -LiteralPath $inventoryPath -NoTypeInformation -Encoding UTF8
$rows | Format-Table -AutoSize
Write-Host "Wrote $inventoryPath"
