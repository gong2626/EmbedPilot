/* Synthetic public demo fixture: deliberate out-of-bounds access. */
int read_sensor(void)
{
    int samples[2] = {1, 2};
    return samples[2];
}
