[CmdletBinding()]
param(
    [string[]]$Dataset = @("all"),
    [string]$SourceRoot = (Join-Path $PSScriptRoot "..\data\source\jangafx\vbt_source_subset"),
    [string]$MatchedRoot = (Join-Path $PSScriptRoot "..\data\source\jangafx\matched_openvdb"),
    [string]$NanoRoot = (Join-Path $PSScriptRoot "..\data\source\jangafx\nanovdb"),
    [string]$ManifestDir = (Join-Path $PSScriptRoot "..\data\source\jangafx\manifests"),
    [string]$VbtCatalog = (Join-Path $PSScriptRoot "..\data\catalog.json"),
    [string]$VbtMetadataRoot = (Join-Path $PSScriptRoot "..\data\smoke"),
    [string]$Converter = (Join-Path $PSScriptRoot "..\3D\vdb_tools\build\Release\vdb_to_nanovdb.exe"),
    [string]$GridName = "density",
    [switch]$KeepNanoVDB,
    [switch]$KeepMatchedOpenVDB
)

$ErrorActionPreference = "Stop"
$sourcePath = (Resolve-Path -LiteralPath $SourceRoot).Path
$converterPath = (Resolve-Path -LiteralPath $Converter).Path
$vbtCatalogData = Get-Content -LiteralPath (Resolve-Path -LiteralPath $VbtCatalog).Path -Raw | ConvertFrom-Json
New-Item -ItemType Directory -Force -Path $MatchedRoot, $NanoRoot, $ManifestDir | Out-Null

function Get-OutputMetric {
    param(
        [Parameter(Mandatory)][string]$Text,
        [Parameter(Mandatory)][string]$Name
    )
    $match = [regex]::Match($Text, [regex]::Escape($Name) + ':\s+([0-9.eE+-]+)')
    if (-not $match.Success) {
        throw "Converter metric '$Name' was not found in output: $Text"
    }
    return [double]::Parse($match.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

$datasetDirs = @(Get-ChildItem -LiteralPath $sourcePath -Directory | Where-Object {
    ($Dataset -contains "all") -or ($Dataset -contains $_.Name)
})
if ($datasetDirs.Count -eq 0) {
    throw "No extracted datasets matched: $($Dataset -join ', ')"
}

$allRows = [System.Collections.Generic.List[object]]::new()
foreach ($datasetDir in $datasetDirs) {
    $allVdbFiles = @(Get-ChildItem -LiteralPath $datasetDir.FullName -Recurse -File -Filter "*.vdb")
    $officialSourceBytes = [long](($allVdbFiles | Measure-Object Length -Sum).Sum)
    $metadataPath = Join-Path (Join-Path $VbtMetadataRoot $datasetDir.Name) ($datasetDir.Name + "_density.metadata.json")
    if (Test-Path -LiteralPath $metadataPath) {
        $vbtMetadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
        $expectedNames = @{}
        foreach ($name in $vbtMetadata.frame_files) { $expectedNames[[string]$name] = $true }
        $vdbFiles = @($allVdbFiles | Where-Object { $expectedNames.ContainsKey($_.Name) })
        if ($vdbFiles.Count -ne $expectedNames.Count) {
            $foundNames = @{}
            foreach ($file in $vdbFiles) { $foundNames[$file.Name] = $true }
            $missing = @($expectedNames.Keys | Where-Object { -not $foundNames.ContainsKey($_) })
            throw "Frame contract mismatch for $($datasetDir.Name): expected=$($expectedNames.Count) found=$($vdbFiles.Count) missing=$($missing -join ', ')"
        }
    } else {
        $vbtMetadata = $null
        $vdbFiles = $allVdbFiles
    }
    $vdbFiles = @($vdbFiles |
        Sort-Object @{ Expression = {
            $match = [regex]::Match($_.BaseName, '(\d+)$')
            if ($match.Success) { [int]$match.Groups[1].Value } else { [int]::MaxValue }
        } }, Name)
    $vbtAsset = @($vbtCatalogData.assets | Where-Object {
        $_.category -eq "smoke" -and $_.dataset -eq $datasetDir.Name -and $_.field -eq $GridName
    } | Select-Object -First 1)
    $vbtBytes = if ($vbtAsset.Count -eq 1) { [long]$vbtAsset[0].bytes } else { 0L }
    if ($vdbFiles.Count -eq 0) {
        Write-Warning "No VDB files found under $($datasetDir.FullName)"
        continue
    }

    $nanoDir = Join-Path $NanoRoot $datasetDir.Name
    $matchedDir = Join-Path $MatchedRoot $datasetDir.Name
    New-Item -ItemType Directory -Force -Path $matchedDir, $nanoDir | Out-Null
    Write-Host "[$($datasetDir.Name)] converting $($vdbFiles.Count) $GridName grids"

    foreach ($vdb in $vdbFiles) {
        $frameMatch = [regex]::Match($vdb.BaseName, '(\d+)$')
        $frame = if ($frameMatch.Success) { [int]$frameMatch.Groups[1].Value } else { -1 }
        $nanoPath = Join-Path $nanoDir ($vdb.BaseName + ".nvdb")
        $nanoZipPath = Join-Path $nanoDir ($vdb.BaseName + ".zip.nvdb")
        $matchedPath = Join-Path $matchedDir ($vdb.BaseName + ".density.vdb")
        $timer = [System.Diagnostics.Stopwatch]::StartNew()
        $output = (& $converterPath --input-vdb $vdb.FullName --output-nvdb $nanoPath --output-nvdb-zip $nanoZipPath --output-matched-vdb $matchedPath --grid-name $GridName 2>&1 | Out-String)
        $exitCode = $LASTEXITCODE
        $timer.Stop()
        if ($exitCode -ne 0 -or -not (Test-Path -LiteralPath $nanoPath) -or -not (Test-Path -LiteralPath $nanoZipPath) -or -not (Test-Path -LiteralPath $matchedPath)) {
            throw "NanoVDB conversion failed for $($vdb.FullName): $output"
        }

        $nanoFile = Get-Item -LiteralPath $nanoPath
        $nanoZipFile = Get-Item -LiteralPath $nanoZipPath
        $matchedFile = Get-Item -LiteralPath $matchedPath
        $bufferBytes = [long](Get-OutputMetric -Text $output -Name "nanovdb-buffer-bytes")
        $allRows.Add([pscustomobject]@{
            dataset = $datasetDir.Name
            frame = $frame
            official_source_frames = $allVdbFiles.Count
            official_source_openvdb_bytes = $officialSourceBytes
            matched_sequence_frames = $vdbFiles.Count
            openvdb_file = $vdb.Name
            source_openvdb_bytes = $vdb.Length
            grid = $GridName
            vbt_density_bytes = $vbtBytes
            matched_openvdb_bytes = $matchedFile.Length
            nanovdb_codec = "NONE"
            nanovdb_buffer_bytes = $bufferBytes
            nanovdb_file_bytes = $nanoFile.Length
            nanovdb_zip_file_bytes = $nanoZipFile.Length
            openvdb_density_read_ms = [math]::Round((Get-OutputMetric -Text $output -Name "openvdb-read-ms"), 3)
            matched_openvdb_write_ms = [math]::Round((Get-OutputMetric -Text $output -Name "matched-vdb-write-ms"), 3)
            nanovdb_convert_ms = [math]::Round((Get-OutputMetric -Text $output -Name "nanovdb-convert-ms"), 3)
            nanovdb_none_write_ms = [math]::Round((Get-OutputMetric -Text $output -Name "nanovdb-none-write-ms"), 3)
            nanovdb_zip_write_ms = [math]::Round((Get-OutputMetric -Text $output -Name "nanovdb-zip-write-ms"), 3)
            baseline_generation_wall_ms = [math]::Round($timer.Elapsed.TotalMilliseconds, 3)
        })

        if (-not $KeepNanoVDB) {
            $resolvedNano = $nanoFile.FullName
            if (-not $resolvedNano.StartsWith((Resolve-Path -LiteralPath $NanoRoot).Path, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "Refusing to remove NanoVDB outside NanoRoot: $resolvedNano"
            }
            Remove-Item -LiteralPath $resolvedNano -Force
            Remove-Item -LiteralPath $nanoZipFile.FullName -Force
        }
        if (-not $KeepMatchedOpenVDB) {
            $resolvedMatched = $matchedFile.FullName
            if (-not $resolvedMatched.StartsWith((Resolve-Path -LiteralPath $MatchedRoot).Path, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "Refusing to remove matched OpenVDB outside MatchedRoot: $resolvedMatched"
            }
            Remove-Item -LiteralPath $resolvedMatched -Force
        }
    }
}

$framesPath = Join-Path $ManifestDir "vdb_nanovdb_frames.csv"
$measuredDatasets = @($datasetDirs | ForEach-Object Name)
$existingRows = if (Test-Path -LiteralPath $framesPath) {
    @(Import-Csv -LiteralPath $framesPath | Where-Object { $measuredDatasets -notcontains $_.dataset })
} else {
    @()
}
$combinedRows = @($existingRows) + @($allRows)
$combinedRows = @($combinedRows | Sort-Object dataset, @{ Expression = { [int]$_.frame } })
$combinedRows | Export-Csv -LiteralPath $framesPath -NoTypeInformation -Encoding UTF8
$summary = @($combinedRows | Group-Object dataset | ForEach-Object {
    $rows = @($_.Group)
    $sourceBytes = [long](($rows | Measure-Object source_openvdb_bytes -Sum).Sum)
    $matchedBytes = [long](($rows | Measure-Object matched_openvdb_bytes -Sum).Sum)
    $nanoBytes = [long](($rows | Measure-Object nanovdb_file_bytes -Sum).Sum)
    $nanoZipBytes = [long](($rows | Measure-Object nanovdb_zip_file_bytes -Sum).Sum)
    $officialFrames = [int]$rows[0].official_source_frames
    $officialBytes = [long]$rows[0].official_source_openvdb_bytes
    $vbtBytes = [long]$rows[0].vbt_density_bytes
    [pscustomobject]@{
        dataset = $_.Name
        matched_frames = $rows.Count
        official_source_frames = $officialFrames
        official_source_openvdb_bytes = $officialBytes
        matched_sequence_source_openvdb_bytes = $sourceBytes
        matched_density_openvdb_bytes = $matchedBytes
        nanovdb_none_bytes = $nanoBytes
        nanovdb_zip_bytes = $nanoZipBytes
        vbt_density_bytes = $vbtBytes
        source_openvdb_over_vbt = if ($vbtBytes -gt 0) { [math]::Round($sourceBytes / $vbtBytes, 6) } else { $null }
        matched_openvdb_over_vbt = if ($vbtBytes -gt 0) { [math]::Round($matchedBytes / $vbtBytes, 6) } else { $null }
        nanovdb_over_vbt = if ($vbtBytes -gt 0) { [math]::Round($nanoBytes / $vbtBytes, 6) } else { $null }
        nanovdb_zip_over_vbt = if ($vbtBytes -gt 0) { [math]::Round($nanoZipBytes / $vbtBytes, 6) } else { $null }
        nanovdb_over_matched_openvdb = [math]::Round($nanoBytes / $matchedBytes, 6)
        nanovdb_zip_over_matched_openvdb = [math]::Round($nanoZipBytes / $matchedBytes, 6)
        mean_source_openvdb_frame_bytes = [math]::Round($sourceBytes / $rows.Count, 3)
        mean_matched_openvdb_frame_bytes = [math]::Round($matchedBytes / $rows.Count, 3)
        mean_nanovdb_frame_bytes = [math]::Round($nanoBytes / $rows.Count, 3)
        mean_nanovdb_zip_frame_bytes = [math]::Round($nanoZipBytes / $rows.Count, 3)
        mean_openvdb_density_read_ms = [math]::Round((($rows | Measure-Object openvdb_density_read_ms -Average).Average), 3)
        mean_matched_openvdb_write_ms = [math]::Round((($rows | Measure-Object matched_openvdb_write_ms -Average).Average), 3)
        mean_nanovdb_convert_ms = [math]::Round((($rows | Measure-Object nanovdb_convert_ms -Average).Average), 3)
        mean_nanovdb_none_write_ms = [math]::Round((($rows | Measure-Object nanovdb_none_write_ms -Average).Average), 3)
        mean_nanovdb_zip_write_ms = [math]::Round((($rows | Measure-Object nanovdb_zip_write_ms -Average).Average), 3)
        failures = 0
    }
})
$summaryPath = Join-Path $ManifestDir "vdb_nanovdb_summary.csv"
$summary | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8
$summary | Format-Table -AutoSize
Write-Host "Wrote $framesPath"
Write-Host "Wrote $summaryPath"
