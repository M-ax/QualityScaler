# Appended to PyInstaller 6.16.0's native splash script at build time.
# JPEG quality stages are embedded at build time; splash_image stays pristine.
# Extraction occupies 0-70%; Python startup phases occupy the remaining range.
# The file count is an estimate, not a byte percentage or time prediction.
set qs_progress 0.0
set qs_target 0.0
set qs_extracted 0
set qs_last_frame -1

image create photo qs_frame -width $image_width -height $image_height
foreach item [.root.canvas find all] {
    if {[.root.canvas type $item] eq "image"} {
        .root.canvas itemconfigure $item -image qs_frame
    }
}

# Four physical image pixels, flush with the bottom edge and above the artwork.
set qs_bar_top [expr {$image_height - 4}]
.root.canvas create rectangle 0 $qs_bar_top $image_width $image_height \
    -fill "#281d2b" -outline ""
set qs_bar [.root.canvas create rectangle 0 $qs_bar_top 0 $image_height \
    -fill "#f35088" -outline ""]

proc qs_render {} {
    global qs_progress qs_quality_images qs_quality_thresholds qs_last_frame
    global qs_bar qs_bar_top
    global image_width image_height
    .root.canvas coords $qs_bar 0 $qs_bar_top \
        [expr {round($image_width * $qs_progress)}] $image_height

    set count [llength $qs_quality_images]
    set index 0
    foreach threshold [lrange $qs_quality_thresholds 1 end] {
        if {$qs_progress < $threshold} {
            break
        }
        incr index
    }
    if {$qs_progress >= 1.0} {
        set index $count
    }
    if {$index != $qs_last_frame} {
        qs_frame blank
        if {$index == $count} {
            # Restore the original exactly, without residual compression artifacts.
            qs_frame copy splash_image -compositingrule set
        } else {
            qs_frame copy [lindex $qs_quality_images $index] -compositingrule set
        }
        set qs_last_frame $index
    }
}

proc qs_tick {} {
    global qs_progress qs_target
    if {$qs_progress < $qs_target} {
        # Ease toward real milestones; never invent progress during a stalled load.
        # Native splash initialization omits Tcl's scripted min/max math functions.
        set step [expr {($qs_target - $qs_progress) * 0.18}]
        if {$step < 0.002} {
            set step 0.002
        }
        set qs_progress [expr {$qs_progress + $step}]
        if {$qs_progress > $qs_target} {
            set qs_progress $qs_target
        }
        qs_render
    }
    if {$qs_progress < 1.0} {
        after 25 qs_tick
    }
}

proc qs_set_target {target} {
    global qs_target
    if {$target > 1.0} {
        set target 1.0
    }
    if {$target > $qs_target} {
        set qs_target $target
    }
}

# The upstream trace calls this procedure for both bootloader filenames and
# pyi_splash.update_text(). Consume our phase prefix instead of displaying it.
proc canvas_text_update {canvas tag _var - -} {
    global qs_target qs_progress qs_extracted qs_extraction_files
    upvar #0 $_var text
    if {[regexp {^QS_PROGRESS:([0-9]+):(.*)$} $text -> progress message]} {
        qs_set_target [expr {$progress / 100.0}]
        if {$qs_target >= 1.0} {
            # Paint the pristine final frame before the desktop's delayed close.
            set qs_progress 1.0
            qs_render
            # Mapping the desktop can cover this otherwise non-topmost window.
            # Keep only the brief completion frame above it, not the whole load.
            wm attributes . -topmost 1
            raise .
        }
    } else {
        set message $text
        incr qs_extracted
        if {$qs_extracted < $qs_extraction_files} {
            qs_set_target [expr {0.70 * $qs_extracted / $qs_extraction_files}]
        } else {
            qs_set_target 0.70
        }
    }
    $canvas itemconfigure $tag -text $message
}

# Render before the native event loop starts, avoiding a flash of the sharp logo.
qs_render
after 25 qs_tick
