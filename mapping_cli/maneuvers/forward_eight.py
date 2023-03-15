from decord import VideoReader

from mapping_cli.maneuvers.maneuver import Maneuver
from mapping_cli.utils import detect_marker


def get_marker_from_frame(frame, marker_list, marker_dict):
    markers = detect_marker(frame, marker_dict)
    if len(markers) > 0:
        permuted_markers = list(
            filter(
                lambda marker: marker in marker_list,
                list(sorted([int(i) for i in markers])),
            )
        )
    else:
        return None
    return permuted_markers


class ForwardEight(Maneuver):
    def run(self) -> None:
        vid = VideoReader(self.inputs["back_video"])
        frames = range(0, len(vid), self.config["skip_frames"])
        marker_list = self.config["marker_list"]
        verification_sequence = self.config["marker_order"]
        markers_detected = []

        for i in frames:
            vid.seek_accurate(i)
            frame = vid.next().asnumpy()
            permuted_markers = get_marker_from_frame(
                frame, marker_list, self.config["marker_dict"]
            )
            if permuted_markers is not None:
                if isinstance(permuted_markers, (list, tuple)):
                    val_markers = []
                    for permuted_marker in permuted_markers:
                        if permuted_marker not in markers_detected:
                            val_markers.append(permuted_marker)

                    if len(val_markers) > 0:
                        markers_detected += val_markers

                elif permuted_markers != markers_detected[-1]:
                    markers_detected += list(permuted_markers)
        sequence_verified = False
        print("Detected: ", markers_detected)

        verified_sequence = None

        for sequence_verification in verification_sequence:
            str_seq = "".join([str(i) for i in sequence_verification])
            iter_seq_verification = iter(str_seq)
            str_markers = "".join([str(i) for i in markers_detected])
            res = all(
                next((ele for ele in iter_seq_verification if ele == chr), None)
                is not None
                for chr in list(str_markers)
            )
            if res and len(markers_detected) >= len(sequence_verification) - 1:
                sequence_verified = True
                verified_sequence = sequence_verification
                break

        print("Verified: ", sequence_verified)
        return (sequence_verified, markers_detected), {}
