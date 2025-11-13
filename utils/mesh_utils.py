import os

def save_landmarks_as_ply(lmk3d, lmk_output_path):
    with open(lmk_output_path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(lmk3d.shape[0]))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        for v in lmk3d:
            f.write('{} {} {}\n'.format(v[0], v[1], v[2]))
    print("Saved 3D landmarks to:", lmk_output_path)


def write_simple_obj( mesh_v, mesh_f, filepath, verbose=False ):
    with open( filepath, 'w') as fp:
        for v in mesh_v:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
        for f in mesh_f+1: # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )
    if verbose:
        print('mesh saved to: ', filepath)
