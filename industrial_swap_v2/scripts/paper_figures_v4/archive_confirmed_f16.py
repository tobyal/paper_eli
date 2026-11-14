"""Archive the confirmed spatial-change F16 without changing its contents."""
import archive_confirmed_f14_f15 as archive


def main():
    archive.FIGURES = {
        'F16': {
            'source': 'F16_spatial_change_analysis',
            'stem': 'figure_F16',
            'title': 'Spatial Change Analysis of Cross-Category Equipment Replacement',
            'description': 'Case00, Case02 and Case07: original equipment, reconfigured scene and visible projected spatial change. Same-camera old-only, shared and new-only regions with actual union-relative percentages.',
            'summary': 'Case00、Case02、Case07 跨类别设备替换；同相机原设备、替换场景与 Old-only / Shared / New-only 可见投影变化。',
            'notes': 'Metrics describe 2D visible projections, not physical 3D volume. Raw masks, float contributions, cameras and source hashes are retained. Guard fallback and the Case07 stored yaw correction remain documented in the unchanged caption and sources.\n',
        }
    }
    archive.main()
    index_path = archive.COLLECTION / 'index.html'
    index = index_path.read_text(encoding='utf-8')
    assert 'F15、F17 为已确认' in index
    archive.write_text(index_path, index.replace('F15、F17 为已确认', 'F15、F16、F17 为已确认', 1))


if __name__ == '__main__':
    main()
