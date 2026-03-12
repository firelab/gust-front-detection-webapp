import dayjs from 'dayjs';
import utc from 'dayjs/plugin/utc';
import timezone from 'dayjs/plugin/timezone';

// Extend the dayjs library with plugins
dayjs.extend(utc);
dayjs.extend(timezone);

export default dayjs;
